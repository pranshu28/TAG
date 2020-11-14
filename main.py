from tag_update import *
from er import *
from agem import *


def train_single_epoch(args, net, optimizer, loader, criterion, task_id=None, manual=False, lr=0.01, buffer=None):
	net = net.to(DEVICE)
	net.train()
	alpha_mean = {}
	# task_cluster = None
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		if task_id is not None:
			pred = net(data, task_id+1)
		else:
			pred = net(data)
		net.zero_grad()

		if buffer is not None:
			if args.opt=='agem':
				net = buffer.observe_agem(net, data, task_id, target)
				continue
			else:
				if task_id > 0:
					mem_x, mem_y, b_task_ids = buffer.sample(args.batch_size, exclude_task=None, pr=False)
					mem_pred = net(mem_x, None)
					mem_pred = apply_mask(mem_y, mem_pred, net.n_classes)
					loss_mem = criterion(mem_pred, mem_y)
					loss_mem.backward()
				buffer.add_reservoir(data, target, None, task_id)

		loss = criterion(pred, target)
		loss.backward()

		if manual:
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


def run(args, train_loaders, test_loaders, val_loaders=None):
	buffer = None
	verbose = False

	acc_db, loss_db = init_experiment(args)
	model = get_benchmark_model(args)

	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	manual = args.opt == 'param'

	if args.opt is not None:
		opt = {'rms': torch.optim.RMSprop, 'adagrad': torch.optim.Adagrad, 'adam': torch.optim.Adam}
		if manual:
			optimizer = tag_opt(model, args, args.tasks, lr=args.lr, optim=args.tag_opt, b=args.b)
		elif args.opt == 'er':
			buffer = ER(args)
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # if args.gamma!=1.0 else 0.0)
		elif args.opt == 'agem':
			optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # if args.gamma!=1.0 else 0.0)
			buffer = AGEM(model, optimizer, criterion, args)
		else:
			optimizer = opt[args.opt](model.parameters(), lr=args.lr)

	continuum = np.random.choice(np.arange(1, args.tasks + 1), 100) if args.multi == 1 else np.arange(1, args.tasks + 1)

	tasks_done = []
	print(continuum)

	for current_task_id in tqdm(continuum):  # range(1, args.tasks+1)
		train_loader = train_loaders[current_task_id-1]# if 'cifar' not in args.dataset else tasks[current_task_id]['train']
		lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)

		best_val_loss, overfit, best_model = 100., 0, None
		for epoch in (range(1, args.epochs_per_task+1)):
			if args.opt is None:
				optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8 if args.gamma != 1.0 else 0.0)

			model, alpha_mean = train_single_epoch(args, model, optimizer, train_loader, criterion, current_task_id-1, manual, lr, buffer)


			if args.epochs_per_task>20 and val_loaders is not None:
				val_loader = val_loaders[current_task_id - 1]
				metrics = eval_single_epoch(model, val_loader, criterion, current_task_id)
				val_loss = metrics['loss']
				if val_loss<best_val_loss:
					best_val_loss, best_model = val_loss, model
					overfit = 0
				else:
					overfit+=1
					if overfit>=5:
						model = best_model
						break

			############ Analysis Part #############
			if verbose:
				mat = np.array([alpha_mean[i] for i in alpha_mean])
				if current_task_id != 1 and alpha_mean != {}:
					imp = np.round(mat.mean(axis=0), 3)
				else:
					imp = [1.0]

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
				save_checkpoint(model, time, manual, prev_task_id, metrics, imp)
		# print("TASK {} / {}".format(current_task_id, args.tasks), '\tAvg Acc:', avg_acc)

	if args.multi != 1:
		# print(acc_db)
		score, forget, learn_acc = end_experiment(args, acc_db, loss_db)
	else:
		score, forget, learn_acc = avg_acc, 0., 0.
	return score, forget, learn_acc


if __name__ == "__main__":
	args = parse_arguments()
	args.device = DEVICE
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	tasks=None
	val_loaders = None

	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	if args.dataset == 'cifar100':
		tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		if args.tasks == 10:
			val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 100
	elif 'imagenet' in args.dataset:
		train_loaders, test_loaders = [CLDataLoader(elem, args, train=t) for elem, t in zip(get_miniimagenet(args), [True, False])]
		args.classes = 100
	else:# args.dataset == 'cub':
		train_loaders, test_loaders = [CLDataLoader(elem, args, train=t) for elem, t in zip(get_split_cub_(args), [True, False])]
		args.classes = 200
	print("loaded all tasks!")

	# all_scores = []
	# for seed in range(args.runs):
	# 	args.seed += seed
	# 	score, forget, learn_acc = run(args, train_loaders, test_loaders, val_loaders)
	# 	all_scores += [[score, forget, learn_acc]]
	# all_scores = np.array(all_scores)
	# print('\nFinal Average accuracy = ',all_scores.mean(axis=0)[0],'+/-',all_scores.std(axis=0)[0],
	#       'forget = ', all_scores.mean(axis=0)[1],'+/-',all_scores.std(axis=0)[1],
	#       'learning accuracy = ', all_scores.mean(axis=0)[2],'+/-',all_scores.std(axis=0)[2] )
	# print('------------------- Experiment ended -----------------\n\n\n')

	lrs = (0.00005, 0.000025, 0.00001)
	args.runs=2
	bs = (3, 5)
	for lr in lrs:
		args.lr = lr*10
		for b in bs:
			args.b = b
			all_scores = []
			for seed in range(args.runs):
				args.seed += seed
				score, forget, learn_acc = run(args, train_loaders, test_loaders)
				all_scores += [[score, forget, learn_acc]]
			all_scores = np.array(all_scores)
			print(lr, b, 'Final Average accuracy = ',all_scores.mean(axis=0)[0],'+/-',all_scores.std(axis=0)[0],
			      'forget = ', all_scores.mean(axis=0)[1],'+/-',all_scores.std(axis=0)[1],
			      'learning accuracy = ', all_scores.mean(axis=0)[2], '+/-', all_scores.std(axis=0)[2])

	# dropouts = (0.0, 0.1, 0.25, 0.5)
	# lrs = (0.005, 0.001, 0.0005)
	# bs = (0.9, 0.8, 0.7, 0.6)
	# args.runs = 2
	# for dropout in dropouts:
	# 	args.dropout = dropout
	# 	for lr in lrs:
	# 		args.lr = lr
	# 		for b in bs:
	# 			args.gamma = b
	# 			all_scores = []
	# 			for seed in range(args.runs):
	# 				args.seed += seed
	# 				score, forget = run(args, train_loaders, test_loaders)
	# 				all_scores += [[score, forget]]
	# 			all_scores = np.array(all_scores)
	# 			print(dropout, lr, b, 'Final Average accuracy = ',all_scores.mean(axis=0)[0],'+/-',all_scores.std(axis=0)[0], 'forget = ', all_scores.mean(axis=0)[1],'+/-',all_scores.std(axis=0)[1])
