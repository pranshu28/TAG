from tag_update import *
from existing_methods.er import *
from existing_methods.agem import *
from existing_methods.ewc import *
from existing_methods.ogd import *


def train_single_epoch(args, net, optimizer, loader, criterion, task_id=None, tag=False, ALGO=None):
	"""
	Run one epoch for the given optimizer/method
	:param args:
	:param net: Model
	:param optimizer: Type of optimizer to be applied
	:param loader: Data loader specific to the given task
	:param criterion: Loss function
	:param task_id: Task identity (assuming the task incremental learning setup)
	:param tag: Indicates whether TAG update needs to applied or not
	:param ALGO: Existing baseline to be applied
	:return: Model with updated parameters and alpha values (from TAG updates)
	"""
	net = net.to(DEVICE)
	net.zero_grad()
	net.train()
	alpha_mean = {}
	for step, (X, Y) in enumerate(loader):
		X = X.to(DEVICE)
		Y = Y.to(DEVICE)
		if task_id is not None:
			pred = net(X, task_id+1)
		else:
			pred = net(X)
		net.zero_grad()

		# EWC / OGD / AGEM / ER
		if ALGO is not None:
			if 'ewc' in args.opt:
				loss_ewc = args.lambd * ALGO.penalty(net)
				loss_ewc.backward()
				torch.nn.utils.clip_grad_norm_(net.parameters(), 100)
			elif 'ogd' in args.opt:
				loss = criterion(pred, Y)
				loss.backward()
				net = ALGO.optimizer_step(optimizer)
				continue
			elif 'agem' in args.opt:
				net = ALGO.observe_agem(net, X, task_id, Y)
			else:
				if task_id > 0:
					mem_x, mem_y, b_task_ids = ALGO.sample(args.batch_size, exclude_task=None, pr=False)
					mem_pred = net(mem_x, None)
					mem_pred = apply_mask(mem_y, mem_pred, net.n_classes)
					loss_mem = criterion(mem_pred, mem_y)
					loss_mem.backward()
				ALGO.add_reservoir(X, Y, None, task_id)

		if 'agem' not in args.opt:
			loss = criterion(pred, Y)
			loss.backward()

		if tag:
			optimizer.step(net, task_id, step)
			if task_id > 0:
				alpha_mean = store_alpha(optimizer, task_id, step, alpha_mean)
		else:
			optimizer.step()
	return net, alpha_mean


def eval_single_epoch(net, loader, criterion, task_id=None):
	"""
	Evaluate the current model on test dataset of the given task_id
	:param net: Current model
	:param loader: Test data loader
	:param criterion: Loss function
	:param task_id: Task identity
	:return:
	"""
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			if task_id is not None:
				output = net(data, task_id)
			else:
				output = net(data)
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}


def avg_runs_exp(runs, validate=False):
	"""
	Get average of the results from multiple runs
	"""
	all_scores = []
	r = 0
	attempts = 0
	while r<runs and attempts<5:
		args.seed += 1
		score, forget, learn_acc = continuum_run(args, train_loaders, val_loaders if validate else test_loaders)
		if score!=0:
			all_scores += [[score, forget, learn_acc]]
			r+=1
			attempts = 0
		attempts += 1
	all_scores = np.array(all_scores)
	if len(all_scores)>0:
		print('\nFinal Average accuracy = ', all_scores.mean(axis=0)[0], '+/-', all_scores.std(axis=0)[0],
		      'forget = ', all_scores.mean(axis=0)[1], '+/-', all_scores.std(axis=0)[1],
		      'learning accuracy = ', all_scores.mean(axis=0)[2], '+/-', all_scores.std(axis=0)[2])
	print('------------------- Experiment ended -----------------\n\n\n')
	return all_scores.mean(axis=0)[0]


def hyp_lr():
	"""
	Grid search over Learning rate
	"""
	lrs = (0.1, 0.05, 0.01, 0.001)
	best_hyp, best_acc = 0, 0
	for lr in lrs:
		args.lr = lr
		print(args.lr)
		args.seed = 0
		acc = avg_runs_exp(args.runs, validate=True)
		if acc > best_acc:
			best_acc = acc
			best_hyp = lr
	print('Best [lr]:', best_hyp)


def hyp_tag():
	"""
	Grid search for TAG: Learning rate and b
	"""
	bs = (5,1,7, 3)
	lrs = (0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001)
	best_hyp, best_acc = 0, 0
	for b in bs:
		args.b = b
		for lr in lrs:
			args.lr = lr
			print(args.lr, args.b)
			args.seed = 0
			acc = avg_runs_exp(args.runs, validate=True)
			if acc > best_acc:
				best_acc = acc
				best_hyp = [lr,b]
	print('Best [lr, b]:', best_hyp)


def hyp_ewc():
	"""
	Grid search for EWC: Learning rate and Lambda
	"""
	ls, lambdas = (0.1, 0.05, 0.01, 0.001), (1, 10, 100)
	best_hyp, best_acc = 0, 0
	for l in ls:
		args.lr =l
		for b in lambdas:
			args.lambd = b
			print(l, b)
			args.seed = 0
			acc = avg_runs_exp(args.runs, validate=True)
			if acc > best_acc:
				best_acc = acc
				best_hyp = [l,b]
	print('Best [lr, lambda]:', best_hyp)


def hyp_stable():
	"""
	Grid search for Stable SGD: Learning rate, decay and Dropout
	"""
	dropouts = (0.0, 0.1, 0.25, 0.5)
	lrs = (0.1, 0.05, 0.01, 0.005, 0.001)
	decays = (0.9, 0.8, 0.7)
	best_hyp, best_acc = 0, 0
	for dropout in dropouts:
		args.dropout = dropout
		for lr in lrs:
			args.lr = lr
			for b in decays:
				args.gamma = b
				print(dropout, lr, b)
				args.seed = 0
				acc = avg_runs_exp(args.runs, validate=True)
				if acc>best_acc:
					best_acc = acc
					best_hyp = [dropout, lr, b]
	print('Best [dropout, lr, decay]:', best_hyp)


def continuum_run(args, train_loaders, test_loaders):
	"""
	Single run for the given dataset
	"""
	ALGO = None

	acc_db, loss_db = init_experiment(args)
	model = get_benchmark_model(args)

	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	tag = 'tag' in args.opt
	optimizer = None

	# Create object of class
	if args.opt != '':
		opt = {'rms': torch.optim.RMSprop, 'adagrad': torch.optim.Adagrad, 'adam': torch.optim.Adam}
		for i in opt:
			if i in args.opt:
				optimizer = opt[i](model.parameters(), lr=args.lr)
				break
		if tag:
			optimizer = TAG(model, args, args.tasks, lr=args.lr, optim=args.tag_opt, b=args.b)
		if optimizer is None:
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
		if 'er' in args.opt:
			ALGO = ER(args)
		if 'agem' in args.opt:
			ALGO = AGEM(model, optimizer, criterion, args)
		if 'ewc' in args.opt:
			ALGO = EWC(model, criterion)
		if 'ogd' in args.opt:
			ALGO = OGD(args, model, optimizer)

	continuum = np.tile(np.arange(1, args.tasks + 1), 6) if args.multi == 1 else np.arange(1, args.tasks + 1)

	tasks_done = []
	print(continuum)

	for current_task_id in (continuum):

		# Naive SGD / Stable SGD
		lr = max(args.lr * (args.gamma ** current_task_id), 0.00005)
		if args.opt == '':
			optimizer = torch.optim.SGD(model.parameters(), lr=lr)

		# Training part
		best_val_loss, overfit = np.inf, 0
		train_loader = train_loaders[current_task_id-1]
		iterator = tqdm(range(1, args.epochs_per_task+1)) if args.epochs_per_task!=1 else range(1, args.epochs_per_task+1)
		for epoch in iterator:
			model, alpha_mean = train_single_epoch(args, model, optimizer, train_loader, criterion, current_task_id-1, tag, ALGO)

			# Early stopping in case of large number of epochs
			if args.epochs_per_task>20 and test_loaders is not None:
				val_loader = val_loaders[current_task_id - 1]
				metrics = eval_single_epoch(model, val_loader, criterion, current_task_id)
				val_loss = metrics['loss']
				if val_loss<best_val_loss:
					best_val_loss = val_loss
					overfit = 0
				else:
					overfit+=1
					if overfit>=5:
						break

			# Collect alphas values for analysis
			alpha_val = [1.0]
			if tag and args.tag_opt=='rms':
				mat = np.array([alpha_mean[i] for i in alpha_mean])
				if current_task_id != 1 and alpha_mean != {}:
					alpha_val = np.round(mat.mean(axis=0), 3)

		if tag:
			optimizer.update_all(current_task_id-1)
		if 'ogd' in args.opt:
			ALGO._update_mem(current_task_id, train_loader)
		if 'ewc' in args.opt:
			loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=200, shuffle=True)
			ALGO.update(model, current_task_id, loader)


		time += 1
		if current_task_id not in tasks_done:
			tasks_done += [current_task_id]

		# Evaluation part
		avg_acc = 0.
		for prev_task_id in tasks_done:  # range(1, current_task_id+1):
			model = model.to(DEVICE)
			test_loader = test_loaders[prev_task_id - 1]
			metrics = eval_single_epoch(model, test_loader, criterion, prev_task_id)
			avg_acc += metrics['accuracy'] / len(tasks_done)
			if args.multi !=1:
				acc_db, loss_db = log_metrics(metrics, time, prev_task_id, acc_db, loss_db)
				if (args.opt == 'tag' and args.tag_opt == 'rms') or args.opt=='rms' and verbose: # verbose
					print_details(tag, prev_task_id, metrics, alpha_val)
		print("TASK {} / {}".format(current_task_id, args.tasks), '\tAvg Acc:', avg_acc)

		torch.cuda.empty_cache()

	if args.multi != 1:
		score, forget, learn_acc = end_experiment(args, acc_db, loss_db)
	else:
		score, forget, learn_acc = avg_acc, 0., 0.
	return score, forget, learn_acc


if __name__ == "__main__":
	args = parse_arguments()
	args.device = DEVICE
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	tasks=None
	verbose = False
	grid_search = args.hyp_gs != ''

	print('CUDA:', torch.cuda.is_available())

	# Get data loaders for the given dataset
	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	if args.dataset in ['cifar100','cifar10']:
		tasks = get_split_cifar100_tasks(args.tasks, args.batch_size, grid_search)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 100
	elif args.dataset == 'imagenet':
		train_loaders, test_loaders, val_loaders = [CLDataLoader(elem, args, train=t)
		                                            for elem, t in zip(get_miniimagenet(args, grid_search), [True, False, False])]
		args.classes = 100
	elif args.dataset == 'cub':
		train_loaders, test_loaders, val_loaders = [CLDataLoader(elem, args, train=t)
		                                            for elem, t in zip(get_split_cub(args, grid_search), [True, False, False])]
		args.classes = 200
	elif args.dataset == '5data':
		tasks = get_5_datasets_tasks(args.tasks, args.batch_size, grid_search)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 50
	else:
		tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		args.classes = 10
	print("loaded all tasks!")

	# Run the experiment for multiple runs
	if not grid_search:
		avg_runs_exp(args.runs)
	else:
		print('\n\n Hyperparameter search:',args.hyp_gs)
		hyp_fun = {'ewc':hyp_ewc, 'tag':hyp_tag, 'stable':hyp_stable, 'lr':hyp_lr}
		hyp_fun[args.hyp_gs]()
