import random
from tag_update import *
from er import *
from agem import *
from ewc import *
from ogd import *


def train_single_epoch(args, net, optimizer, loader, criterion, task_id=None, tag=False, lr=0.01, ALGO=None):
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
			if args.opt == 'ewc':
				loss_ewc = args.lambd * ALGO.penalty(net)
				loss_ewc.backward()
			elif args.opt == 'ogd':
				loss = criterion(pred, target)
				loss.backward()
				net = ALGO.optimizer_step(net, lr, task_id, batch_idx, optimizer)
				continue
			elif args.opt=='agem':
				net = ALGO.observe_agem(net, data, task_id, target)
				continue
			else:
				if task_id > 0:
					mem_x, mem_y, b_task_ids = ALGO.sample(args.batch_size, exclude_task=None, pr=False)
					mem_pred = net(mem_x, None)
					mem_pred = apply_mask(mem_y, mem_pred, net.n_classes)
					loss_mem = criterion(mem_pred, mem_y)
					loss_mem.backward()
				ALGO.add_reservoir(data, target, None, task_id)

		loss = criterion(pred, target)
		loss.backward()
		if tag:
			optimizer.step(net, task_id, batch_idx, lr=lr)
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


def avg_runs_exp(runs):
	all_scores = []
	r=0
	while r<runs:
		args.seed += 1
		score, forget, learn_acc = continuum_run(args, train_loaders, test_loaders, val_loaders)
		if score!=0:
			all_scores += [[score, forget, learn_acc]]
			r+=1
	all_scores = np.array(all_scores)
	print('\nFinal Average accuracy = ', all_scores.mean(axis=0)[0], '+/-', all_scores.std(axis=0)[0],
	      'forget = ', all_scores.mean(axis=0)[1], '+/-', all_scores.std(axis=0)[1],
	      'learning accuracy = ', all_scores.mean(axis=0)[2], '+/-', all_scores.std(axis=0)[2])
	print('------------------- Experiment ended -----------------\n\n\n')


def hyp_tag(lrs, runs):
	bs = (3, 5)
	for lr in lrs:
		args.lr = lr
		for b in bs:
			args.b = b
			print(args.lr, args.b)
			avg_runs_exp(runs)


def hyp_ogd(ls, bs):
	args.runs=2
	for b in bs:
		args.batch_size = b
		for l in ls:
			args.epochs_per_task =l
			print(b, l)
			avg_runs_exp(args.runs)


def hyp_ewc(ls, bs):
	args.runs=1
	for l in ls:
		args.lr =l
		for b in bs:
			args.lambd = b
			print(l, b)
			avg_runs_exp(args.runs)


def hyp_stable():
	dropouts = (0.0, 0.1, 0.25, 0.5)
	lrs = (0.05, 0.01)#, 0.005)
	bs = (0.9, 0.8, 0.7, 0.6)
	args.runs = 1
	for dropout in dropouts:
		args.dropout = dropout
		for lr in lrs:
			args.lr = lr
			for b in bs:
				args.gamma = b
				print(dropout, lr, b)
				avg_runs_exp(args.runs)


def continuum_run(args, train_loaders, test_loaders, val_loaders=None):
	ALGO = None

	acc_db, loss_db = init_experiment(args)
	model = get_benchmark_model(args)

	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	tag = args.opt == 'param'

	if args.opt is not None:
		opt = {'rms': torch.optim.RMSprop, 'adagrad': torch.optim.Adagrad, 'adam': torch.optim.Adam}
		if tag:
			optimizer = tag_opt(model, args, args.tasks, lr=args.lr, optim=args.tag_opt, b=args.b)
		elif args.opt == 'er':
			ALGO = ER(args)
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # if args.gamma!=1.0 else 0.0)
		elif args.opt == 'agem':
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # if args.gamma!=1.0 else 0.0)
			ALGO = AGEM(model, optimizer, criterion, args)
		elif args.opt == 'ewc':
			sample_size = 200
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
		elif args.opt == 'ogd':
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
			ALGO = OGD(args, model, optimizer)
		else:
			optimizer = opt[args.opt](model.parameters(), lr=args.lr)

	continuum = np.tile(np.arange(1, args.tasks + 1), 5) if args.multi == 1 else np.arange(1, args.tasks + 1)

	tasks_done = []
	print(continuum)
	skip = 0

	for current_task_id in (continuum):  # range(1, args.tasks+1)
		train_loader = train_loaders[current_task_id-1]
		lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)

		if args.opt == 'ewc' and current_task_id!=1:
			old_tasks = []
			for sub_task in range(current_task_id-1):
				loader = torch.utils.data.DataLoader(train_loaders[sub_task].dataset, batch_size=sample_size, num_workers=0, shuffle=False)
				old_tasks += next(iter(loader))[0]
			old_tasks = random.sample(old_tasks, k=sample_size)
			ALGO = EWC(model, old_tasks)

		best_val_loss, overfit = np.inf, 0
		iterator = tqdm(range(1, args.epochs_per_task+1)) if args.epochs_per_task!=1 else range(1, args.epochs_per_task+1)

		for epoch in iterator:
			if args.opt is None:
				optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8 if args.gamma != 1.0 else 0.0)

			model, alpha_mean = train_single_epoch(args, model, optimizer, train_loader, criterion, current_task_id-1, tag, lr, ALGO)

			if args.epochs_per_task>20 and val_loaders is not None:
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

			############ Analysis Part #############
			imp = [1.0]
			if verbose and tag:
				mat = np.array([alpha_mean[i] for i in alpha_mean])
				if current_task_id != 1 and alpha_mean != {}:
					imp = np.round(mat.mean(axis=0), 3)

		if tag:
			optimizer.update_all(current_task_id-1)
		elif args.opt=='ogd':
			ALGO._update_mem(current_task_id, train_loader)

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
				if verbose:
					save_checkpoint(model, time, tag, prev_task_id, metrics, imp)
		print("TASK {} / {}".format(current_task_id, args.tasks), '\tAvg Acc:', avg_acc)
		if avg_acc<=20:
			skip=1
			break

		torch.cuda.empty_cache()
	if args.multi != 1:
		if skip==1:
			print('Aborting this run!!')
			return 0., 0., 0.
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
	val_loaders = None
	print('CUDA:', torch.cuda.is_available())

	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	if args.dataset in ['cifar100','cifar10']:
		tasks = get_split_cifar100_tasks(args.tasks, args.batch_size)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		if args.tasks == 10:
			val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 100
	elif args.dataset == 'imagenet':
		train_loaders, test_loaders = [CLDataLoader(elem, args, train=t) for elem, t in zip(get_miniimagenet(args), [True, False])]
		args.classes = 100
	elif args.dataset == 'cub':
		train_loaders, test_loaders = [CLDataLoader(elem, args, train=t) for elem, t in zip(get_split_cub_(args), [True, False])]
		args.classes = 200
	elif args.dataset == '5data':
		tasks = get_5_datasets_tasks(args.tasks, args.batch_size)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 50
	else:
		tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		args.classes = 10
	print("loaded all tasks!")

	verbose = False
	# args.seed = 5
	# score, forget, learn_acc = continuum_run(args, train_loaders, test_loaders, val_loaders)
	avg_runs_exp(args.runs)
	# hyp_ewc([0.1, 0.05, 0.01, 0.001], [50,100,200])
	# hyp_ogd([1, 10, 20], [32,64, 256])
	# hyp_stable()
	# lrs = (0.00005, 0.000025, 0.00001)    #  (0.0005, 0.0025, 0.001, 0.005)
	# hyp_tag(lrs,1)
