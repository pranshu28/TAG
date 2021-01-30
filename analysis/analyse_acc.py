import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def get_naive(data):
	return [data[:,0],data[:,1], data[:,2],data[:,3]]


def taged_plot():
	data1 = np.array([[ 51.36 , 3.21 ],[ 62.79 , 0.29 ],[ 51.22 , 2.82 ],[ 62.55 , 0.31 ],[ 54.25 , 2.0 ],[ 64.8 , 1.01 ],[ 59.14 , 1.77 ],[ 66.56 , 0.39 ]])
	data2 = np.array([[ 48.19 , 0.79 ],[ 57.2 , 1.37 ],[ 49.01 , 0.78 ],[ 57.65 , 0.72 ],[ 50.32 , 1.29 ],[ 58.96 , 0.89 ],[ 52.76 , 1.53 ],[ 58.91 , 1.11 ]])
	data3 = np.array([[ 54.88 , 1.83 ],[ 61.58 , 1.24 ],[ 54.36 , 1.33 ],[ 61.32 , 0.61 ],[ 56.91 , 1.37 ],[ 62.93 , 0.93 ],[ 59.25 , 0.82 ],[ 67.47 , 0.69 ]])
	data4 = np.array([[ 46.48 , 3.62 ],
[ 60.8 , 4.32 ],
[ 50.24 , 3.41 ],
[ 64.77 , 1.05 ],
[ 54.38 , 2.1 ],
[ 69.87 , 2.88 ],
[ 57.45 , 2.99 ],
[ 69.96 , 1.23 ]])
	ls = ['Naive SGD', 'TAG-RMSProp', 'EWC', 'TAG-EWC', 'A-GEM_1', 'TAG-A-GEM_1','ER_1', 'TAG-ER_1']
	dataset = {'Split-CIFAR100':data1, 'Split-miniImageNet':data2, 'Split-CUB':data3, '5-dataset':data4}
	plt.figure(figsize=(9,4))
	plt.ylim(40,75)
	cs =  plt.cm.tab20(np.linspace(0, 1, 20))
	for i,data in enumerate(dataset):
		for j, method in enumerate(ls):
			method = method.replace('_1', '')
			if 'TAG-' in method and method != 'TAG-RMSProp':
				method = 'TAGed ' + method[4:]
			t = dataset[data][j,0]
			x = [5*i+(0.5*j)]
			err = dataset[data][j,1]
			if j%2!=0:
				plt.bar(x, t, yerr=err, width = 0.5, label=method, hatch = '/', edgecolor='black', color=cs[j])
			else:
				plt.bar(x, t, yerr=err, width=0.5, label=method, edgecolor='black', color=cs[j])
	# plt.legend(bbox_to_anchor=(1, 1))
	plt.ylabel(('Accuracy (%)'))
	plt.xticks([1.5, 6.5, 11.5, 16.5], dataset)


def naive_plot(f=''):
	data1 = np.array([[ 51.36 , 3.21 , 0.18 , 0.03 ],[ 50.98 , 1.05 , 0.22 , 0.02 ],[ 63.22 , 0.78 , 0.1 , 0.01 ],[ 48.91 , 2.88 , 0.2 , 0.03 ],[ 62.79 , 0.29 , 0.1 , 0.01 ],[ 48.69 , 1.56 , 0.19 , 0.01 ],[62.086 , 1.3698, 0.0981 , 0.00750]])
	data2 = np.array([[ 48.19 , 0.79 , 0.13 , 0.01 ],
[ 49.93 , 1.8 , 0.16 , 0.01 ],
[ 56.99 , 0.57 , 0.05 , 0.004 ],
[ 45.06 , 0.6 , 0.21 , 0.01 ],
[ 57.2 , 1.37 , 0.06 , 0.02 ],
[ 45.92 , 1.58 , 0.2 , 0.02 ],
[ 57.61 , 1.24 , 0.06 , 0.01 ]])
	data3 = np.array([[ 54.88 , 1.83 , 0.12 , 0.01 ],
[ 55.44 , 1.22 , 0.1 , 0.0 ],
[ 61.76 , 1.04 , 0.11 , 0.01 ],
[ 49.4 , 1.77 , 0.24 , 0.01 ],
[ 61.58 , 1.24 , 0.11 , 0.01 ],
[ 44.21 , 1.98 , 0.25 , 0.02 ],
[ 57.54 , 0.83 , 0.1 , 0.0 ]])
	data4 = np.array([[ 46.48 , 3.62 , 0.48 , 0.05 ],
[ 54.23 , 2.96 , 0.39 , 0.04 ],
[ 63.98 , 2.15 , 0.27 , 0.02 ],
[ 45.49 , 1.89 , 0.5 , 0.03 ],
[ 60.8 , 4.32 , 0.32 , 0.05 ],
[ 49.37 , 3.57 , 0.48 , 0.04 ],
[ 64.25 , 2.65 , 0.29 , 0.03 ]])

	labels = ['Naive SGD', 'Naive Adagrad', 'TAG-Adagrad', 'Naive RMSProp', 'TAG-RMSProp', 'Naive Adam', 'TAG-Adam']
	dataset = ['Split-CIFAR100', 'Split-miniImageNet', 'Split-CUB', '5-dataset']

	data = {}
	for di, d in enumerate(np.array([get_naive(data1), get_naive(data2), get_naive(data3), get_naive(data4)])):
		print(d)
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
	# plt.ylim(0,100)
	# cs = ['tab:brown','tab:blue','tab:blue','tab:orange','tab:orange','tab:green','tab:green']
	cs =  plt.cm.tab20(np.linspace(0, 1, 20))
	for i,o in enumerate(ls):
		t = [data[d+f][o] for d in dataset]
		x = [5*o+(0.5*i) for o in range(len(dataset))]
		err = [data['e_'+d+f][o] for d in dataset]
		if i%2==0 and i>0:
			plt.bar(x, t, yerr=err, width = 0.5, label = o, hatch = '/', edgecolor='black', color=cs[i+5])
		else:
			plt.bar(x, t, yerr=err, width=0.5, label=o, edgecolor='black', color=cs[i+5])
	# plt.legend()#bbox_to_anchor=(1.001, 1))
	plt.ylabel(('Accuracy (%)' if f=='' else 'Forgetting'))
	plt.xticks([1.5, 6.5, 11.5, 16.5], dataset)

def acc(dataset):
	f = open(dataset+'.txt', 'r')
	lines = f.readlines()
	dataset = dataset.replace('_5','')
	dataset = dataset.replace('_10','')
	# dataset = dataset.replace('_','')
	print(dataset)
	curr = 0

	continuum = np.zeros(20)
	tasks = np.arange(1,21)

	opt_data , final_res = {}, {}
	curr_opt = 'Naive SGD'
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
		elif line[:4]=='Abor':
			continuum = np.zeros(20)
			runs-=1
		elif line[:4]=='----':
			curr=0
			runs+=1
			try:
				if runs>0 and np.sum(continuum)!=0.0:
					opt_data[curr_opt] += [continuum.copy()]
			except:
				opt_data[curr_opt] = [continuum]
				continuum = np.zeros(20)
		elif 'mem' in line:
			curr_opt = curr_opt+'_'+l[-1].split('=')[-1]


	def plot_means(f, dataset, exp, tasks, data, standard_dev, c, style = '-'):
		plt.figure(f, figsize=(6,5))
		inds = [0,4,9,14,19] if dataset in ['cifar','imagenet','cub'] else range(5)
		if 'TAG-' in exp and exp!='TAG-RMSProp':
			exp = 'TAGed '+exp[4:]
		plt.plot(tasks[inds], data[inds], color=c, label=exp.replace('_1', ''), marker='o', linestyle=style, linewidth=2)
		plt.fill_between(tasks[inds], data[inds] - standard_dev[inds], data[inds] + standard_dev[inds],color=c, alpha=0.1)
		fs = 10
		plt.yticks(fontsize=fs)
		plt.xticks(tasks[inds], fontsize=fs)
		plt.title(dataset)
		# plt.ylim(35,75)
		plt.xlabel('Tasks', fontsize=fs)
		plt.ylabel('Accuracy (%)', fontsize=fs)
		plt.legend()#bbox_to_anchor=(1.01, 1))

	dataset = {'rotate_eq': 'Rotated MNIST (30)','rotate':'Rotated MNIST','permute':'Permute MNIST','cifar10_resnet':'CIFAR-100 (10 tasks)','cifar10':'CIFAR-100 (10 tasks)','cifar':'Split-CIFAR100', 'imagenet':'Split-miniImageNet', 'cub':'Split-CUB', '5data':'5-dataset'}[dataset]

	# ls = ['Naive SGD', 'Naive Adagrad', 'TAG-Adagrad', 'Naive RMSProp','TAG-RMSProp',  'Naive Adam', 'TAG-Adam']
	ls = ['Naive SGD', 'Naive RMSProp', 'EWC', 'A-GEM_1', 'ER_1', 'Stable SGD', 'TAG-RMSProp']
	# ls = ['Naive SGD', 'TAG-RMSProp', 'EWC', 'TAG-EWC', 'A-GEM_1', 'TAG-A-GEM_1','ER_1', 'TAG-ER_1']
	# ls = [ 'A-GEM_1', 'A-GEM_2', 'A-GEM_3', 'A-GEM_5','A-GEM_10', 'ER_1', 'ER_2', 'ER_3', 'ER_5', 'ER_10', 'TAG-RMSProp']

	colors = plt.cm.Dark2(np.linspace(0, 1, len(ls)))
	for ind, i in enumerate(range(len(ls))): #[0,2,4,6]):
		exp = ls[i]
		if exp not in final_res:
			continue
		content = np.array(final_res[exp].split(' '))[[5,7,11,13,18,20]].astype(float).round(2)
		plot_means(1, dataset, exp, tasks, opt_data[exp], opt_data[exp+'_std'], colors[ind], style='-')
		# print('[',content[0], ',', content[1] , ',', content[2], ',', content[3],'],') #
		print(exp,'\n\t\t& $', content[0], '~(\pm', content[1], ')$ & $', content[2], '~(\pm', content[3], ')$ & $', content[4], '~(\pm', content[5], ')$')

		# print('\t\t', content[0], '(±', content[1], ')', content[2], '(±', content[3], ')', content[4], '(±', content[5], ')')


# acc('cifar')
# acc('imagenet')
# acc('cub')
acc('5data_5')

# naive_plot('')
# naive_plot('_f')

# taged_plot()

# plt.show()
