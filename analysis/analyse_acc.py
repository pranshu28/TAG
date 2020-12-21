import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def naive_plot(f=''):
	d1 = [ 52.2,50.9,47.5,52.0,63.6,63.3,61.6]
	d1_err = [3.8, 3.0, 2.1, 1.1, 1.3, 0.9, 0.5]
	d2 = [ 51.4,42.9,45.8, 49.5,55.5,57.9,55.6]
	d2_err = [1.0, 2.0, 2.9, 1.3, 1.4, 0.8, 0.9]
	d3 = [ 55.8,49.9,44.8,55.9,61.9,61.8,58.7]
	d3_err = [1.0, 1.8, 3.2, 1.8, 1.3, 0.9, 1.7]
	d4 = [ 52.47,42.8,48.27,45.03,61.7, 60.3,64.18]
	d4_err = [4.22, 2.43, 3.51, 1.56, 2.25, 2.08, 2.22]
	f1 = [ 0.2,0.17,0.2,0.18,0.09,0.09,0.09]
	f1_err = [0.04, 0.03, 0.02, 0.01, 0.01, 0.004, 0.006]
	f2 = [ 0.14,0.19,0.2,0.13,0.11,0.06,0.08]
	f2_err = [0.01, 0.01, 0.03, 0.01, 0.007, 0.01, 0.008]
	f3 = [ 0.11,0.23,0.25,0.12,0.1,0.11,0.1]
	f3_err = [0.005, 0.02, 0.03, 0.02, 0.01, 0.005, 0.01]
	f4 = [0.4, 0.49, 0.49, 0.46, 0.29, 0.31, 0.28]
	f4_err = [0.05, 0.04, 0.04, 0.03, 0.027, 0.02, 0.03]

	labels = ['Naive Adagrad', 'Naive RMSProp', 'Naive Adam', 'Naive SGD', 'TAG-Adagrad', 'TAG-RMSProp', 'TAG-Adam']
	dataset = ['Split-CIFAR100', 'Split-miniImageNet', 'Split-CUB', '5-dataset']

	data = {}
	for di, d in enumerate([[d1, d1_err, f1, f1_err], [d2, d2_err, f2, f2_err], [d3, d3_err, f3, f3_err], [d4, d4_err, f4, f4_err]]):
		data[dataset[int(di)]] = {}
		data[dataset[int(di)]+'_f'] = {}
		data['e_'+dataset[int(di)]] = {}
		data['e_'+dataset[int(di)]+'_f'] = {}
		for i,k in enumerate(labels):
			data[dataset[int(di)]][k] = d[0][i]
			data['e_'+dataset[int(di)]][k] = d[1][i]
			data[dataset[int(di)]+'_f'][k] = d[2][i]
			data['e_'+dataset[int(di)]+'_f'][k] = d[3][i]

	ls = ['Naive SGD', 'Naive Adagrad', 'TAG-Adagrad', 'Naive RMSProp', 'TAG-RMSProp', 'Naive Adam', 'TAG-Adam']
	plt.figure(figsize=(7,4))
	# cs = ['tab:brown','tab:blue','tab:blue','tab:orange','tab:orange','tab:green','tab:green']
	cs =  plt.cm.tab20(np.linspace(0, 1, 20))
	print(data)
	for i,o in enumerate(ls):
		t = [data[d+f][o] for d in dataset]
		x = [5*o+(0.5*i) for o in range(len(dataset))]
		err = [data['e_'+d+f][o] for d in dataset]
		if i%2==0 and i>0:
			plt.bar(x, t, yerr=err, width = 0.5, label = o, hatch = '/', edgecolor='black', color=cs[i+5])
		else:
			plt.bar(x, t, yerr=err, width=0.5, label=o, edgecolor='black', color=cs[i+5])
	# plt.legend(bbox_to_anchor=(1.001, 1))
	plt.ylabel(('Accuracy (%)' if f=='' else 'Forgetting'))
	plt.xticks([1.5, 6.5, 11.5, 16.5], dataset)

def acc():
	for dataset in ['permute']:# ['rotate','permute','cifar','imagenet','cub','5data']:  # 'cifar_5', 'imagenet_5','cub_5','5data_5'
		print('\n\n',dataset)
		f = open(dataset+'.txt', 'r')
		lines = f.readlines()
		dataset = dataset.replace('_5','')
		curr = 0

		continuum = np.zeros(20)
		tasks = np.arange(1,21)

		opt_data , final_res = {}, {}
		curr_opt = 'Plastic (Naive) SGD'
		runs = -1
		for i, line in enumerate(lines):
			l = line.strip().split('\t')
			while '' in l:
				l.remove('')
			if line[:4]=='Fina':
				final_res[curr_opt] = line.strip()
			if '>' in line or i==len(lines)-1:
				if runs>0 or i==len(lines)-1:
					arr = np.array(opt_data[curr_opt])
					opt_data[curr_opt] = arr.sum(axis=0)/runs
					opt_data[curr_opt+'_std'] = arr.std(axis=0)
				try:
					curr_opt = l[0].replace('>', '').strip()
					opt_data[curr_opt] = np.zeros(20)
					runs = -1
				except:
					pass
			elif line[:4]=='TASK':
				continuum[curr] = float(l[-1].split(' ')[-1])
				curr += 1
			elif line[:4]=='----':
				curr=0
				runs+=1
				try:
					if runs>0:
						opt_data[curr_opt] += [continuum.copy()]
				except:
					opt_data[curr_opt] = [continuum]
					continuum = np.zeros(20)
			elif 'mem' in line:
				curr_opt = curr_opt+'_'+l[-1].split('=')[-1]


		def plot_means(f, dataset, exp, tasks, data, standard_dev, c):
			plt.figure(f, figsize=(5,4))
			inds = [0,4,9,14,19]#
			plt.plot(tasks[inds], data[inds], color=c, label=exp.replace('_1', ''), marker='o', linewidth=2)
			plt.fill_between(tasks[inds], data[inds] - standard_dev[inds], data[inds] + standard_dev[inds],color=c, alpha=0.2)
			# plt.ylim(0,100)
			fs = 10
			plt.yticks(fontsize=fs)
			plt.xticks(tasks[inds], fontsize=fs)
			plt.title(dataset)
			plt.xlabel('Tasks', fontsize=fs)
			plt.ylabel('Accuracy (%)', fontsize=fs)
			plt.legend(fontsize=8)#bbox_to_anchor=(1.01, 1))

		dataset = {'rotate':'Rotated MNIST','permute':'Permute MNIST','cifar10_resnet':'CIFAR-100 (10 tasks)','cifar10':'CIFAR-100 (10 tasks)','cifar':'Split-CIFAR100', 'imagenet':'Split-miniImageNet', 'cub':'Split-CUB', '5data':'5-dataset'}[dataset]

		# ls = ['Plastic (Naive) SGD', 'Plastic (Naive) RMSProp', 'Plastic (Naive) Adagrad', 'Plastic (Naive) Adam', 'Manual Adagrad (Ours)', 'Manual RMSProp (Ours)', 'Manual Adam (Ours)']
		# ls = ['Naive Adagrad', 'Naive RMSProp', 'Naive Adam', 'Naive SGD', 'TAG-Adagrad', 'TAG-RMSProp', 'TAG-Adam']

		# ls = ['Plastic (Naive) SGD', 'Plastic (Naive) RMSProp', 'A-GEM_1', 'ER_1', 'Stable SGD', 'Manual RMSProp (Ours)']
		ls = ['Naive SGD', 'Naive RMSProp', 'A-GEM_1', 'ER_1', 'Stable SGD', 'TAG-RMSProp']
		colors = plt.cm.Dark2(np.linspace(0, 1, len(ls)))
		for i,exp in enumerate(ls):
			print(exp)
			try:
				content = np.array(final_res[exp].split(' '))[[5,7,11,13,18,20]].astype(float).round(2)
				plot_means(1, dataset, exp, tasks, opt_data[exp], opt_data[exp+'_std'], colors[i])
				print('\t\t', content[0], '(+/-', content[1], ')', content[2], '(+/-', content[3], ')', content[4], '(+/-', content[5], ')')
				print('\t\t$', content[0], '~(\pm', content[1], ')$ & $', content[2], '~(\pm', content[3], ')$ & $', content[4], '~(\pm', content[5], ')$')
			except:
				try:
					content = np.array(final_res[exp].split(' '))[[5,7,11,13]].astype(float).round(2)
					plot_means(1, dataset, exp, tasks, opt_data[exp], opt_data[exp+'_std'], colors[i])
					print('\t\t', content[0], '(+/-', content[1], ')', content[2], '(+/-', content[3], ')')
					print('\t\t$', content[0], '~(\pm', content[1], ')$ & $', content[2], '~(\pm', content[3], ')$')
				except:
					pass
	print('\n\n ')

acc()
# naive_plot('')
# naive_plot('_f')
plt.show()
