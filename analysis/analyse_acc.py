import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def get_naive(data):
	return [data[:,0],data[:,1], data[:,2],data[:,3]]


def taged_plot():
	data1 = np.array([[ 51.358 , 3.214 ],
[ 48.906 , 2.88 ],
[ 62.786 , 0.286 ],
[ 49.058 , 3.438 ],
[ 51.482 , 2.057 ],
[ 63.977 , 0.655 ],
[ 54.248 , 2.001 ],
[ 49.53 , 2.164 ],
[ 64.8 , 1.007 ],
[ 59.144 , 1.769 ],
[ 49.706 , 0.782 ],
[ 66.564 , 0.391 ]])
	data2 = np.array([[ 48.647 , 1.482 ],
[ 45.065 , 0.597 ],
[ 57.202 , 1.368 ],
[ 47.87 , 2.081 ],
[ 42.403 , 2.991 ],
[ 58.146 , 0.163 ],
[ 50.322 , 1.286 ],
[ 44.155 , 1.709 ],
[ 58.962 , 0.886 ],
[ 54.668 , 0.709 ],
[ 43.47 , 0.916 ],
[ 58.913 , 1.111 ]])
	data3 = np.array([[ 54.878 , 1.826 ],
[ 49.401 , 1.775 ],
[ 61.576 , 1.236 ],
[ 55.659 , 0.967 ],
[ 50.894 , 1.891 ],
[ 59.496 , 0.709 ],
[ 56.913 , 1.369 ],
[ 51.512 , 1.639 ],
[ 62.93 , 0.935 ],
[ 59.247 , 0.816 ],
[ 60.027 , 1.079 ],
[ 67.472 , 0.691 ]
])
	data4 = np.array([[ 46.482 , 3.62 ],
[ 45.49 , 1.894 ],
[ 62.591 , 1.815 ],
[ 48.58 , 1.467 ],
[ 57.785 , 2.672 ],
[ 65.667 , 2.48 ],
[ 55.9 , 2.581 ],
[ 57.856 , 2.402 ],
[ 69.866 , 2.876 ],
[ 61.581 , 2.65 ],
[ 52.115 , 2.17 ],
[ 69.957 , 1.232 ]])
	# ls = ['Naive SGD', 'TAG-RMSProp', 'EWC', 'TAG-EWC', 'A-GEM_1', 'TAG-A-GEM_1','ER_1', 'TAG-ER_1']
	ls = ['Naive SGD', 'Naive RMSProp', 'TAG-RMSProp', 'EWC', 'RMSProp EWC', 'TAG-EWC', 'A-GEM_1', 'RMSProp A-GEM_1', 'TAG-A-GEM_1','ER_1', 'RMSProp ER_1', 'TAG-ER_1']
	shift = len(ls)-4
	dataset = {'Split-CIFAR100':data1, 'Split-miniImageNet':data2, 'Split-CUB':data3, '5-dataset':data4}
	plt.figure(figsize=(9,4))
	plt.ylim(35,75)
	# cs =  plt.cm.tab20(np.linspace(0, 1, 20))
	cs = plt.cm.tab20c( (4./3*np.arange(20*3/4)).astype(int))
	for i,data in enumerate(dataset):
		print(data)
		for j, method in enumerate(ls):
			method = method.replace('_1', '')
			if 'TAG-' in method and method != 'TAG-RMSProp':
				method = 'TAGed ' + method[4:]
			t = dataset[data][j,0]
			x = [shift*i+(0.5*j)]
			err = dataset[data][j,1]
			if j%3==2:
				print(method, dataset[data][j,0]-dataset[data][j-2,0], dataset[data][j,0]-dataset[data][j-1,0])
			# if j%2!=0:
			# 	plt.bar(x, t, yerr=err, width = 0.5, label=method, hatch = '/', edgecolor='black', color=cs[j])
			# else:
			plt.bar(x, t, yerr=err, width=0.5, label=method, edgecolor='black', color=cs[j])
		# break
	# plt.legend(bbox_to_anchor=(1, 1))
	plt.ylabel(('Accuracy (%)'))
	# plt.xticks([1.5, shift + 1.5,2*shift+ 1.5, 3*shift+1.5], dataset)
	plt.xticks([2.75, shift + 2.75,2*shift+ 2.75, 3*shift+2.75], dataset)


def naive_plot(f=''):
	data1 = np.array([[ 51.36 , 3.21 , 0.18 , 0.03 ],[ 50.98 , 1.05 , 0.22 , 0.02 ],[ 63.22 , 0.78 , 0.1 , 0.01 ],[ 48.91 , 2.88 , 0.2 , 0.03 ],[ 62.79 , 0.29 , 0.1 , 0.01 ],[ 48.69 , 1.56 , 0.19 , 0.01 ],[62.086 , 1.3698, 0.0981 , 0.00750]])
	data2 = np.array([[ 48.19 , 0.79 , 0.13 , 0.01 ],
[ 49.93 , 1.8 , 0.16 , 0.01 ],
[ 56.99 , 0.57 , 0.05 , 0.004 ],
[ 45.06 , 0.6 , 0.21 , 0.01 ],
[ 57.2 , 1.37 , 0.06 , 0.02 ],
[ 45.92 , 1.58 , 0.2 , 0.02 ],
[ 57.61 , 1.24 , 0.06 , 0.01 ]])
	data3 = np.array([[ 54.878 , 1.826 , 0.122 , 0.014 ],
[ 55.443 , 1.216 , 0.105 , 0.003 ],
[ 61.765 , 1.044 , 0.11 , 0.011 ],
[ 49.401 , 1.775 , 0.237 , 0.01 ],
[ 61.576 , 1.236 , 0.11 , 0.015 ],
[ 44.205 , 1.985 , 0.252 , 0.021 ],
[ 57.538 , 0.83 , 0.1 , 0.004 ]])
	data4 = np.array([[ 46.482 , 3.62 , 0.476 , 0.046 ],
[ 54.234 , 2.961 , 0.392 , 0.039 ],
[ 67.726 , 0.561 , 0.227 , 0.01 ],
[ 45.49 , 1.894 , 0.501 , 0.033 ],
[ 62.591 , 1.815 , 0.294 , 0.022 ],
[ 49.374 , 3.571 , 0.478 , 0.038 ],
[ 63.761 , 2.972 , 0.28 , 0.037 ]])

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
		# if i%2==0 and i>0:
		# 	plt.bar(x, t, yerr=err, width = 0.5, label = o, hatch = '/', edgecolor='black', color=cs[i+5])
		# else:
		plt.bar(x, t, yerr=err, width=0.5, label=o, edgecolor='black', color=cs[i+1])
	fs=10
	# plt.legend(bbox_to_anchor=(1.001, 1))
	plt.yticks(fontsize=15)
	plt.ylabel(('Accuracy (%)' if f=='' else 'Forgetting'), fontsize=15)
	plt.xticks([1.5, 6.5, 11.5, 16.5], dataset, fontsize=10)

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
		fs = 15
		plt.yticks(fontsize=fs)
		plt.xticks(tasks[inds], fontsize=fs)
		plt.title(dataset)
		# plt.ylim(25,60)
		plt.xlabel('Tasks', fontsize=fs)
		plt.ylabel('Accuracy (%)', fontsize=fs)
		plt.legend(fontsize=fs-4)#bbox_to_anchor=(1.01, 1))

	dataset = {'rotate_eq': 'Rotated MNIST (30)','rotate':'Rotated MNIST','permute':'Permute MNIST','cifar10_resnet':'CIFAR-100 (10 tasks)','cifar10':'CIFAR-100 (10 tasks)','cifar':'Split-CIFAR100', 'imagenet':'Split-miniImageNet', 'cub':'Split-CUB', '5data':'5-dataset'}[dataset]

	# ls = ['Naive SGD', 'Naive Adagrad', 'TAG-Adagrad', 'Naive RMSProp','TAG-RMSProp',  'Naive Adam', 'TAG-Adam']
	ls = ['Naive SGD', 'Naive RMSProp', 'EWC', 'A-GEM_1', 'ER_1', 'Stable SGD', 'TAG-RMSProp']
	# ls = ['Naive SGD', 'TAG-RMSProp', 'EWC', 'TAG-EWC', 'A-GEM_1', 'TAG-A-GEM_1','ER_1', 'TAG-ER_1']
	# ls = ['Naive SGD', 'Naive RMSProp', 'TAG-RMSProp', 'EWC', 'RMSProp EWC', 'TAG-EWC', 'A-GEM_1', 'RMSProp A-GEM_1', 'TAG-A-GEM_1','ER_1', 'RMSProp ER_1', 'TAG-ER_1']
	# ls = [ 'A-GEM_1', 'A-GEM_2', 'A-GEM_3', 'A-GEM_5','A-GEM_10', 'ER_1', 'ER_2', 'ER_3', 'ER_5', 'ER_10', 'TAG-RMSProp']

	colors = plt.cm.tab10(np.linspace(0, 1, len(ls)))
	for ind, i in enumerate(range(len(ls))): #[0,2,4,6]):
		exp = ls[i]
		if exp not in final_res:
			continue
		content = []
		for y in final_res[exp].split(' '):
			try:
				content += [np.round(float(y), 2)]
			except:
				pass
		plot_means(1, dataset, exp, tasks, opt_data[exp], opt_data[exp+'_std'], colors[ind], style='-')
		# print('[',content[0], ',', content[1] ,'],') #',', content[2], ',', content[3],
		print(exp,'\n\t\t& $', content[0], '~(\pm', content[1], ')$ & $', content[2], '~(\pm', content[3], ')$ & $', content[4], '~(\pm', content[5], ')$')

		# print('\t\t', content[0], '(±', content[1], ')', content[2], '(±', content[3], ')', content[4], '(±', content[5], ')')


# acc('cifar')
# acc('imagenet')
# acc('cub')
# acc('5data')

# naive_plot('')
naive_plot('_f')

# taged_plot()

plt.show()
