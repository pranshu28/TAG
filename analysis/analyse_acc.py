import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

dataset = 'cub'
f = open(dataset+'.txt', 'r')
lines = f.readlines()
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
			opt_data[curr_opt] /= runs
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
				opt_data[curr_opt] += continuum
		except:
			opt_data[curr_opt] = continuum
			continuum = np.zeros(20)


def plot_means(f, dataset, exp, tasks, data, c):
	plt.figure(f, figsize=(5,4))
	inds = [0,4,9,14,19]
	plt.plot(tasks[inds], data[inds], color=c, label=exp, marker='o', linewidth=2)
	# plt.ylim(0,100)
	fs = 10
	plt.yticks(fontsize=fs)
	plt.xticks(tasks[inds], fontsize=fs)
	plt.title(dataset)
	plt.xlabel('Tasks', fontsize=fs)
	plt.ylabel('Test Accuracy (%)', fontsize=fs)
	plt.legend(fontsize=8)#bbox_to_anchor=(1.01, 1))

dataset = {'cifar':'CIFAR-100', 'imagenet':'Mini-imagenet', 'cub':'CUB'}[dataset]

ls = ['Plastic (Naive) SGD', 'Plastic (Naive) RMSProp', 'A-GEM', 'ER', 'Stable SGD', 'Manual RMSProp (Ours)']
colors = plt.cm.Dark2(np.linspace(0, 1, len(ls)))
for i,exp in enumerate(ls):
	label = exp.replace('Plastic (Naive)','Naive').replace('Manual ','TAG-').replace(' (Ours)','')
	try:
		content = np.array(final_res[exp].split(' '))[[5,7,11,13]].astype(float).round(2)
		print(label,'\n\t\t$',content[0],'~(\pm',content[1],')$ & $',content[2],'~(\pm',content[3],')$')
		plot_means(1, dataset, label, tasks, opt_data[exp], colors[i])
	except:
		pass
plt.show()
