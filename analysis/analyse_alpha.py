import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

dataset = 'cub'
f = open(dataset+'.txt', 'r')
dataset = {'cifar':'CIFAR-100', 'imagenet':'Mini-imagenet', 'cub':'CUB'}[dataset]

lines = f.readlines()
curr = 0
valid = 0
man_acc, rms_acc = np.zeros((20,20)), np.zeros((20,20))
tasks = np.arange(1,21)
corr = np.zeros((20,20))
runs = -1
for i, line in enumerate(lines):
	l = line.split('\t')
	while '' in l:
		l.remove('')
	if '>' in line:
		if 'Manual RMSProp' in line:
			valid = 2
			print(valid)
		elif 'Plastic (Naive) RMSProp' in line:
			valid = 1
			print(valid)
		else:
			valid = 0
	if valid>0:
		if '>' in line:
			if runs > 0 or i == len(lines) - 1:
				man_acc_main /= runs
				if valid==1:
					rms_acc_prev /= runs
				corr_main /= runs
			try:
				man_acc_main = np.zeros((20, 20))
				corr_main = np.zeros((20, 20))
				if valid==1:
					rms_acc_prev = np.zeros((20, 20))
				runs = -1
			except:
				pass
		if line[:4]=='----':
			curr=0
			runs+=1
			try:
				if valid==2:
					man_acc_main += man_acc
					corr_main += corr
				elif valid == 1:
					rms_acc_prev += rms_acc
			except:
				if valid==2:
					man_acc_main = man_acc
					corr_main = corr
					man_acc = np.zeros((20, 20))
					corr = np.zeros((20, 20))
				elif valid == 1:
					rms_acc_prev = rms_acc
					rms_acc = np.zeros((20, 20))
			continue
		if line[:4]=='TASK':
			curr += 1#int(line.split(' ')[1])
		if l[0][:4] == 'Prev':
			prev = int(l[0].split(':')[-1].strip())-1
			if valid == 2:
				corr[prev, curr] = float(l[-1].strip())
				man_acc[prev, curr] = float(l[-2].strip())
			elif valid == 1:
				rms_acc[prev, curr] = float(l[-1].strip())
man_acc_main /= runs
rms_acc_prev /= runs
corr_main /= runs


colors = plt.cm.tab20(np.linspace(0, 1, len(corr)))
corr_main[corr_main<=0.1]=np.nan
corr_main = pd.DataFrame(corr_main)

def plot(f, man_acc, label, i=-1, alpha=1.0):
	man_acc[man_acc == 0] = np.nan
	man_acc = pd.DataFrame(man_acc)
	plt.figure(f)
	# plt.plot(tasks[1:], man_acc.mean(axis=0)[1:]/100, alpha=alpha, label=label)
	if i!=-1:
		plt.plot(tasks[1:], man_acc.ix[i][1:]/100, label=label)


def plot_detailed(lim=5):
	alpha=0.2
	for i in corr_main:
		if i+1 not in lim:
			continue
		plt.figure(i+1)
		plot(i+1,man_acc_main, 'TAG-RMSProp', i, alpha)
		plot(i+1,rms_acc_prev, 'Naive RMSProp', i, alpha)
		plt.xticks(tasks)
		# vals = (corr_main.ix[i][1:]-corr_main.min(axis=0)[1:])/(corr_main.max(axis=0)[1:]-corr_main.min(axis=0)[1:])
		# plt.plot(tasks[1:], vals*0.2+0.8, color=colors[i], label=r'$\alpha(t, '+str(i+1)+')$', marker='o')
		plt.plot(tasks[1:], corr_main.ix[i][1:], color=colors[i], label=r'$\alpha(t, '+str(i+1)+')$', marker='o')
		plt.plot(tasks[1:], corr_main.mean(axis=0)[1:], color = 'black',  alpha=alpha, label=r'$E_{\tau}~~[\alpha(t, \tau)]$')
		plt.plot(tasks[1:], corr_main.max(axis=0)[1:], color = 'black', linestyle=':', alpha=alpha, label=r'$\max_{\tau}~~\alpha(t, \tau)$')
		plt.plot(tasks[1:], corr_main.min(axis=0)[1:], color = 'black', linestyle=':', alpha=alpha, label=r'$\min_{\tau}~~\alpha(t, \tau)$')

		plt.xlabel('Tasks (t)')
		plt.title(r'$\tau=$'+str(i+1)+' ('+dataset+')')
		plt.ylabel('Test Accuracy\t\t\t' + r'$\alpha(t,\tau)$')
		plt.legend()

def plot_means(man_acc_main, rms_acc_prev, corr_main):
	# plot(10,man_acc_main, 'TAG-RMSProp')
	# plot(10,rms_acc_prev, 'Naive RMSProp')
	print(runs)
	print(corr)
	plt.xticks(tasks)
	for i in corr_main:
		print(corr_main.ix[i][1:])
		plt.scatter(tasks[1:], corr_main.ix[i][1:], color=colors[i], label=r'$\alpha(t, '+str(i+1)+')$')
	plt.plot(tasks[1:], corr_main.mean(axis=0)[1:], color = 'black', label=r'$E_{\tau}~~[\alpha(t, \tau)]$')
	plt.plot(tasks[1:], corr_main.max(axis=0)[1:], color = 'black', linestyle=':',label=r'$\max_{\tau}~~\alpha(t, \tau)$')
	plt.plot(tasks[1:], corr_main.min(axis=0)[1:], color = 'black', linestyle=':',label=r'$\min_{\tau}~~\alpha(t, \tau)$')
	plt.xlabel('Tasks (t)')
	plt.title(dataset)
	plt.ylabel('Test Accuracy\t\t\t\t'+r'$\alpha(t,\tau)$')
	plt.legend(bbox_to_anchor=(1.01, 1))

	for i in range(len(corr)):
		plt.figure(2)
		plt.plot(tasks[i:], man_acc_main[i,i:], color=colors[i], label=r'$\tau='+str(i+1)+'$', marker='x')
		plt.figure(3)
		plt.plot(tasks[i:], rms_acc_prev[i,i:], color=colors[i], label=r'$\tau='+str(i+1)+'$', marker='x')

	plt.figure(2)
	man_acc_main[man_acc_main == 0] = np.nan
	man_acc_main = pd.DataFrame(man_acc_main)
	plt.plot(tasks, man_acc_main.mean(axis=0), color='black', label='Mean')
	plt.ylim(0,100)
	plt.xticks(tasks)
	plt.title('TAG-RMSProp ('+dataset+')')
	plt.xlabel('Tasks (t)')
	plt.ylabel('Test Accuracy (%)')
	plt.legend(bbox_to_anchor=(1.01, 1))

	plt.figure(3)
	rms_acc_prev[rms_acc_prev == 0] = np.nan
	rms_acc_prev = pd.DataFrame(rms_acc_prev)
	plt.plot(tasks, rms_acc_prev.mean(axis=0), color='black', label='Mean')
	plt.ylim(0,100)
	plt.xticks(tasks)
	plt.title('Naive RMSProp ('+dataset+')')
	plt.xlabel('Tasks (t)')
	plt.ylabel('Test Accuracy (%)')
	plt.legend(bbox_to_anchor=(1.01, 1))

# plot_means(man_acc_main, rms_acc_prev, corr_main)
plot_detailed(range(10))
plt.show()
