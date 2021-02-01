import random
from tag_update import *
from er import *
from agem import *
from ewc import *
from ogd import *


def train_single_epoch(args, net, optimizer, loader, criterion, task_id=None, tag=False, ALGO=None):
	net = net.to(DEVICE)
	net.zero_grad()
	net.train()
	alpha_mean = {}
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		if task_id is not None:
			pred = net(data, task_id+1)
		else:
			pred = net(data)
		net.zero_grad()

		###### EWC / OGD / AGEM / ER ######
		if ALGO is not None:
			if 'ewc' in args.opt:
				loss_ewc = args.lambd * ALGO.penalty(net)
				loss_ewc.backward()
				torch.nn.utils.clip_grad_norm_(net.parameters(), 100)
			elif 'ogd' in args.opt:
				loss = criterion(pred, target)
				loss.backward()
				net = ALGO.optimizer_step(optimizer)
				continue
			elif 'agem' in args.opt:
				net = ALGO.observe_agem(net, data, task_id, target)
			else:
				if task_id > 0:
					mem_x, mem_y, b_task_ids = ALGO.sample(args.batch_size, exclude_task=None, pr=False)
					mem_pred = net(mem_x, None)
					mem_pred = apply_mask(mem_y, mem_pred, net.n_classes)
					loss_mem = criterion(mem_pred, mem_y)
					loss_mem.backward()
				ALGO.add_reservoir(data, target, None, task_id)

		if 'agem' not in args.opt:
			loss = criterion(pred, target)
			loss.backward()

		if tag:
			optimizer.step(net, task_id, batch_idx)
			if task_id > 0:
				alpha_mean = store_alpha(optimizer, task_id, batch_idx, alpha_mean)
		else:
			optimizer.step()
	return net, alpha_mean


def eval_single_epoch(net, loader, criterion, task_id=None):
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
	lrs = (0.1, 0.05, 0.01, 0.001)
	best_hyp, best_acc = 0, 0
	for lr in lrs:
		args.lr = lr
		print(args.lr)
		acc = avg_runs_exp(args.runs, validate=True)
		if acc > best_acc:
			best_acc = acc
			best_hyp = lr
	print('Best [lr, b]:', best_hyp)


def hyp_tag():
	lrs = (0.00005, 0.000025, 0.00001)
	bs = (1, 3, 5, 7)
	best_hyp, best_acc = 0, []
	for lr in lrs:
		args.lr = lr
		for b in bs:
			args.b = b
			print(args.lr, args.b)
			acc = avg_runs_exp(args.runs, validate=True)
			if acc > best_acc:
				best_acc = acc
				best_hyp = [lr,b]
	print('Best [lr, b]:', best_hyp)


def hyp_ewc():
	ls, bs = (0.1, 0.05, 0.01, 0.001), (1, 10, 100)
	best_hyp, best_acc = 0, []
	for l in ls:
		args.lr =l
		for b in bs:
			args.lambd = b
			print(l, b)
			acc = avg_runs_exp(args.runs, validate=True)
			if acc > best_acc:
				best_acc = acc
				best_hyp = [l,b]
	print('Best [lr, lambd]:', best_hyp)


def hyp_stable():
	dropouts = (0.0, 0.1, 0.25, 0.5)
	lrs = (0.1, 0.05, 0.01, 0.005, 0.001)
	bs = (0.9, 0.8, 0.7)
	best_hyp, best_acc = 0, []
	for dropout in dropouts:
		args.dropout = dropout
		for lr in lrs:
			args.lr = lr
			for b in bs:
				args.gamma = b
				print(dropout, lr, b)
				acc = avg_runs_exp(args.runs, validate=True)
				if acc>best_acc:
					best_acc = acc
					best_hyp = [dropout, lr, b]
	print('Best [dropout, lr, b]:', best_hyp)


def continuum_run(args, train_loaders, test_loaders):
	ALGO = None

	acc_db, loss_db = init_experiment(args)
	model = get_benchmark_model(args)

	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	tag = 'tag' in args.opt

	if args.opt != '':
		opt = {'rms': torch.optim.RMSprop, 'adagrad': torch.optim.Adagrad, 'adam': torch.optim.Adam}
		if args.opt in opt:
			optimizer = opt[args.opt](model.parameters(), lr=args.lr)
		if tag:
			optimizer = tag_opt(model, args, args.tasks, lr=args.lr, optim=args.tag_opt, b=args.b)
		if 'er' in args.opt:
			ALGO = ER(args)
			if not tag:
				optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
		if 'agem' in args.opt:
			if not tag:
				optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
			ALGO = AGEM(model, optimizer, criterion, args)
		if 'ewc' in args.opt:
			sample_size = 200
			ALGO = EWC(model, criterion)
			if not tag:
				optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
		if 'ogd' in args.opt:
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
			ALGO = OGD(args, model, optimizer)

	continuum = np.tile(np.arange(1, args.tasks + 1), 6) if args.multi == 1 else np.arange(1, args.tasks + 1)

	tasks_done = []
	print(continuum)
	skip = 0

	for current_task_id in (continuum):  # range(1, args.tasks+1)
		train_loader = train_loaders[current_task_id-1]
		lr = max(args.lr * (args.gamma ** current_task_id), 0.00005)

		# best_val_loss, overfit = np.inf, 0
		iterator = tqdm(range(1, args.epochs_per_task+1)) if args.epochs_per_task!=1 else range(1, args.epochs_per_task+1)

		for epoch in iterator:
			if args.opt == '':
				optimizer = torch.optim.SGD(model.parameters(), lr=lr)

			model, alpha_mean = train_single_epoch(args, model, optimizer, train_loader, criterion, current_task_id-1, tag, ALGO)

			# if args.epochs_per_task>20 and test_loaders is not None:
			# 	val_loader = val_loaders[current_task_id - 1]
			# 	metrics = eval_single_epoch(model, val_loader, criterion, current_task_id)
			# 	val_loss = metrics['loss']
			# 	if val_loss<best_val_loss:
			# 		best_val_loss = val_loss
			# 		overfit = 0
			# 	else:
			# 		overfit+=1
			# 		if overfit>=5:
			# 			break

			############ Analysis Part #############
			imp = [1.0]
			if tag and args.tag_opt=='rms':
				mat = np.array([alpha_mean[i] for i in alpha_mean])
				if current_task_id != 1 and alpha_mean != {}:
					imp = np.round(mat.mean(axis=0), 3)

		if tag:
			optimizer.update_all(current_task_id-1)
		if 'ogd' in args.opt:
			ALGO._update_mem(current_task_id, train_loader)
		if 'ewc' in args.opt:
			loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=sample_size, shuffle=True)
			ALGO.update(model, current_task_id, loader)


		time += 1
		if current_task_id not in tasks_done:
			tasks_done += [current_task_id]

		avg_acc = 0.
		for prev_task_id in tasks_done:  # range(1, current_task_id+1):
			model = model.to(DEVICE)
			test_loader = test_loaders[prev_task_id - 1]
			metrics = eval_single_epoch(model, test_loader, criterion, prev_task_id)
			avg_acc += metrics['accuracy'] / len(tasks_done)
			if args.multi !=1:
				acc_db, loss_db = log_metrics(metrics, time, prev_task_id, acc_db, loss_db)
				if (args.tag_opt == 'tag' and args.tag_opt == 'rms') or args.opt=='rms': # verbose
					save_checkpoint(model, time, tag, prev_task_id, metrics, imp)
		print("TASK {} / {}".format(current_task_id, args.tasks), '\tAvg Acc:', avg_acc)
		# if avg_acc<=20:
		# 	skip=1
		# 	break

		torch.cuda.empty_cache()
	if args.multi != 1:
		# if skip==1:
		# 	print('Aborting this run!!')
		# 	return 0., 0., 0.
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
	get_val = args.hyp_gs != ''

	print('CUDA:', torch.cuda.is_available())
	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	if args.dataset in ['cifar100','cifar10']:
		tasks = get_split_cifar100_tasks(args.tasks, args.batch_size, get_val)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 100
	elif args.dataset == 'imagenet':
		train_loaders, test_loaders, val_loaders = [CLDataLoader(elem, args, train=t) for elem, t in zip(get_miniimagenet(args, get_val), [True, False, False])]
		args.classes = 100
	elif args.dataset == 'cub':
		train_loaders, test_loaders, val_loaders = [CLDataLoader(elem, args, train=t) for elem, t in zip(get_split_cub_(args, get_val), [True, False, False])]
		args.classes = 200
	elif args.dataset == '5data':
		tasks = get_5_datasets_tasks(args.tasks, args.batch_size, get_val)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 50
	else:
		tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		args.classes = 10
	print("loaded all tasks!")
	verbose = False
	if not get_val:
		avg_runs_exp(args.runs)
	else:
		print('\n\nHyperparameter search',args.hyp_gs)
		hyp_fun = {'ewc':hyp_ewc, 'tag':hyp_tag, 'stable':hyp_stable, 'lr':hyp_lr}
		hyp_fun[args.hyp_gs]()
