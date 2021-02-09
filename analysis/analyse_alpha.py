import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

dataset = 'imagenet'
f = open(dataset + '.txt', 'r')

dataset_name = {'rotate_eq': 'Rotated MNIST (30)', 'rotate': 'Rotated MNIST', 'permute': 'Permute MNIST',
                'cifar10_resnet': 'CIFAR-100 (10 tasks)',
                'cifar10': 'CIFAR-100 (10 tasks)', 'cifar': 'Split-CIFAR100', 'imagenet': 'Split-miniImageNet',
                'cub': 'Split-CUB', '5data': '5-dataset'}[dataset]
ls = ['Naive SGD', 'Naive RMSProp', 'A-GEM', 'ER', 'Stable SGD', 'TAG-RMSProp']

n_tasks = 5 if dataset=='5data' else 20
lines = f.readlines()
curr = 0
valid = 2
man_acc, rms_acc = np.zeros((n_tasks, n_tasks)), np.zeros((n_tasks, n_tasks))
tasks = np.arange(1, n_tasks + 1)
corr = np.zeros((n_tasks, n_tasks))
runs = -1

la = []

for i, line in enumerate(lines):
	l = line.split('\t')
	while '' in l:
		l.remove('')
	if '>' in line:
		if 'TAG-RMSProp' in line:
			valid = 2
			print(valid)
		elif 'Naive RMSProp' in line:
			valid = 1
			print(valid)
		else:
			valid = 0
	if valid > 0:
		# if '>' in line:
		# 	# if runs > 0 or i == len(lines) - 1:
		# 	# 	man_acc_main /= runs
		# 	# 	if valid==1:
		# 	# 		rms_acc_prev /= runs
		# 	# 	corr_main /= runs
		# 	try:
		# 		man_acc_main = np.zeros((n_tasks, n_tasks))
		# 		corr_main = np.zeros((n_tasks, n_tasks))
		# 		if valid == 1:
		# 			rms_acc_prev = np.zeros((n_tasks, n_tasks))
		# 		runs = -1
		# 	except:
		# 		pass
		if line[:4] == '----':
			curr = 0
			runs += 1
			try:
				if valid == 2:
					man_acc_main += [man_acc.copy()]
					corr_main += [corr.copy()]
				elif valid == 1:
					la += [np.mean(np.diag(rms_acc))]
					rms_acc_prev += [rms_acc.copy()]
			except:
				if valid == 2:
					man_acc_main = []
					corr_main = []
					man_acc = np.zeros((n_tasks, n_tasks))
					corr = np.zeros((n_tasks, n_tasks))
				elif valid == 1:
					rms_acc_prev = []
					rms_acc = np.zeros((n_tasks, n_tasks))
			continue
		if line[:4] == 'TASK':
			curr += 1  # int(line.split(' ')[1])
		if l[0][:4] == 'Prev':
			prev = int(l[0].split(':')[-1].strip()) - 1
			if valid == 2:
				corr[prev, curr] = float(l[-1].strip())
				man_acc[prev, curr] = float(l[-2].strip())
			elif valid == 1:
				rms_acc[prev, curr] = float(l[-1].strip())
corr_main_full = np.array(corr_main)
man_acc_main_full = np.array(man_acc_main)
rms_acc_prev_full = np.array(rms_acc_prev)

man_acc_main = man_acc_main_full.mean(axis=0)
man_acc_main_std = man_acc_main_full.std(axis=0)
rms_acc_prev = rms_acc_prev_full.mean(axis=0)
rms_acc_prev_std = rms_acc_prev_full.std(axis=0)
corr_main = corr_main_full.mean(axis=0)
corr_main_std = corr_main_full.std(axis=0)
print(man_acc_main, man_acc_main_std)

# print(' & $',np.mean(la[1:]).round(2),' ~(\pm ',np.std(la[1:]).round(2),' )$ ')
colors = plt.cm.tab20(np.linspace(0, 1, len(corr)))
corr_main[corr_main <= 0.1] = np.nan
corr_main_std[corr_main_std <= 0.0] = np.nan
corr_main = pd.DataFrame(corr_main)
corr_main_std = pd.DataFrame(corr_main_std)


def plot(f, man_acc, std, label, i=-1, alpha=1.0):
	man_acc[man_acc <= 0.0] = np.nan
	man_acc = pd.DataFrame(man_acc) / 100
	std[std <= 0.0] = np.nan
	std = pd.DataFrame(std) / 100
	plt.figure(f)
	# plt.plot(tasks[1:], man_acc.mean(axis=0)[1:]/100, alpha=alpha, label=label)
	if i != -1:
		if i!=0:
			plt.plot(tasks[1:], man_acc.loc[i][1:], label=label)
			plt.fill_between(tasks[1:], man_acc.loc[i][1:] - std.loc[i][1:], man_acc.loc[i][1:] + std.loc[i][1:], alpha=alpha)
		else:
			plt.plot(tasks, man_acc.loc[i], label=label)
			plt.fill_between(tasks, man_acc.loc[i] - std.loc[i], man_acc.loc[i] + std.loc[i], alpha=alpha)


def plot_detailed(lim=5):
	alpha = 0.2
	for i in corr_main:
		if i + 1 not in lim:
			continue
		plt.figure(i + 1)
		plot(i + 1, man_acc_main, man_acc_main_std, 'TAG-RMSProp', i, alpha)
		plot(i + 1, rms_acc_prev, rms_acc_prev_std, 'Naive RMSProp', i, alpha)
		# vals = (corr_main.ix[i][1:]-corr_main.min(axis=0)[1:])/(corr_main.max(axis=0)[1:]-corr_main.min(axis=0)[1:])
		# plt.plot(tasks[1:], vals*0.2+0.8, color=colors[i], label=r'$\alpha(t, '+str(i+1)+')$', marker='o')
		plt.plot(tasks[1:], corr_main.loc[i][1:], color=colors[i], label=r'$\alpha(t, ' + str(i + 1) + ')$', marker='o')
		# plt.fill_between(tasks[1:], corr_main.loc[i][1:] - corr_main_std.loc[i][1:], corr_main.loc[i][1:] + corr_main_std.loc[i][1:], color=colors[i], alpha=0.25)
		plt.plot(tasks[1:], corr_main.mean(axis=0)[1:], color='black', alpha=alpha,
		         label=r'$\mathbb{E}_{\tau}~~[\alpha(t, \tau)]$')
		plt.plot(tasks[1:], corr_main.max(axis=0)[1:], color='black', linestyle=':', alpha=alpha,
		         label=r'$\max_{\tau}~~\alpha(t, \tau)$')
		plt.plot(tasks[1:], corr_main.min(axis=0)[1:], color='black', linestyle=':', alpha=alpha,
		         label=r'$\min_{\tau}~~\alpha(t, \tau)$')
		fs = 15
		plt.xlabel('Tasks (t)')
		plt.yticks(fontsize=fs)
		plt.xticks(tasks, fontsize=fs-4)
		# plt.title(r'$\tau=$' + str(i + 1) + ' (' + dataset_name + ')', fontsize=fs)
		plt.ylabel('Accuracy\t\t\t' + r'$\alpha(t,\tau)$', fontsize=fs)
		plt.legend(fontsize=fs-4)
		plt.savefig('pics/'+dataset_name + '_' + str(i + 1) + '.png')


def plot_means(man_acc_main, rms_acc_prev, corr_main):
	# plot(10,man_acc_main, 'TAG-RMSProp')
	# plot(10,rms_acc_prev, 'Naive RMSProp')
	print(runs)
	print(corr)
	for i in corr_main:
		plt.scatter(tasks[1:], corr_main.ix[i][1:], color=colors[i], label=r'$\alpha(t, ' + str(i + 1) + ')$')
	plt.plot(tasks[1:], corr_main.mean(axis=0)[1:], color='black', label=r'$E_{\tau}~~[\alpha(t, \tau)]$')
	plt.plot(tasks[1:], corr_main.max(axis=0)[1:], color='black', linestyle=':',
	         label=r'$\max_{\tau}~~\alpha(t, \tau)$')
	plt.plot(tasks[1:], corr_main.min(axis=0)[1:], color='black', linestyle=':',
	         label=r'$\min_{\tau}~~\alpha(t, \tau)$')
	if 'rotate' in dataset:
		plt.xticks(tasks, [str(i) + '\n(' + str((i - 1) * 30) + ')' for i in tasks])
		plt.xlabel('Tasks (Degrees)')
	else:
		plt.xticks(tasks)
		plt.xlabel('Tasks (t)')
	plt.title(dataset_name)
	plt.ylabel(r'$\alpha(t,\tau)$')
	plt.legend()  # bbox_to_anchor=(1.01, 1))

	for i in range(len(corr)):
		plt.figure(2)
		plt.plot(tasks[i:], man_acc_main[i, i:], color=colors[i], label=r'$\tau=' + str(i + 1) + '$', marker='x')
		plt.figure(3)
		plt.plot(tasks[i:], rms_acc_prev[i, i:], color=colors[i], label=r'$\tau=' + str(i + 1) + '$', marker='x')

	plt.figure(2)
	man_acc_main[man_acc_main == 0] = np.nan
	man_acc_main = pd.DataFrame(man_acc_main)
	plt.plot(tasks, man_acc_main.mean(axis=0), color='black', label='Mean')
	# plt.ylim(40,100)
	if 'rotate' in dataset:
		plt.xticks(tasks, [str(i) + '\n(' + str((i - 1) * 30) + ')' for i in tasks])
		plt.xlabel('Tasks (Degrees)')
	else:
		plt.xticks(tasks)
		plt.xlabel('Tasks (t)')
	plt.title(r'TAG-RMSProp ($\alpha=1$) (' + dataset_name + ')')
	plt.ylabel('Accuracy (%)')
	plt.legend()  # bbox_to_anchor=(1.01, 1))

	plt.figure(3)
	rms_acc_prev[rms_acc_prev == 0] = np.nan
	rms_acc_prev = pd.DataFrame(rms_acc_prev)
	plt.plot(tasks, rms_acc_prev.mean(axis=0), color='black', label='Mean')
	plt.ylim(40, 100)
	if 'rotate' in dataset:
		plt.xticks(tasks, [str(i) + '\n(' + str((i - 1) * 30) + ')' for i in tasks])
		plt.xlabel('Tasks (Degrees)')
	else:
		plt.xticks(tasks)
		plt.xlabel('Tasks (t)')
	plt.title('Naive RMSProp (' + dataset_name + ')')
	plt.ylabel('Accuracy (%)')
	plt.legend()  # bbox_to_anchor=(1.01, 1))


# plot_means(man_acc_main, rms_acc_prev, corr_main)
plot_detailed(range(min(n_tasks-1,10)))
# plt.show()
