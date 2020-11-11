import math
from utils import *


class manual_opt():
	def __init__(self, model, args, num_tasks, optim='rms', lr=None, b=5):
		self.optim = optim
		self.args = args
		self.iters = 0
		self.model = model
		self.b = b
		self.weight_decay = 0.0
		if self.optim=='adam':
			self.beta1, self.beta2 = 0.9,  0.999
		else:
			self.beta1, self.beta2 = 0.9,  0.99
		self.lr = lr
		self.alpha_add_ = {}
		self.v, self.v_t = {}, {}
		self.m, self.m_t = {}, {}
		for task in range(num_tasks):
			self.v_t[task] = {}
			self.m_t[task] = {}
			self.alpha_add_[task] = {}
			for (name, param) in model.named_parameters():
				if task == 0:
					self.v[name] = torch.zeros_like(param).to(args.device)
					self.m[name] = torch.zeros_like(param).to(args.device)
				self.alpha_add_[task][name] = np.array([1])
				self.v_t[task][name] = torch.zeros_like(param).to(args.device)
				self.m_t[task][name] = torch.zeros_like(param).to(args.device)

	def zero_grad(self):
		return self.model.zero_grad()

	def manual_update_naive(self, name, dw, task, param, lr=None):
		if self.optim=='rms':
			self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * dw ** 2
		else:
			self.v[name] += dw ** 2
		denom = torch.sqrt(self.v[name]) + 1e-8
		return - (lr * dw / denom)

	def manual_update(self, name, dw, task, lr=None):
		bias_corr1, bias_corr2 = 1, 1
		eq = {1:'n,nh->h', 2:'n,nhw->hw', 3:'n,nhwc->hwc', 4: 'n,nhwvd->hwvd', 5:'n,nhwzxc->hwzxc'}[len(dw.shape)]
		new_v = None
		self.m_t[task][name] = self.beta1 * self.m_t[task][name] + (1 - self.beta1) * dw

		if self.optim=='adam':
			bias_corr1, bias_corr2 = 1 - self.beta1 ** (self.iters + 1), 1 - self.beta2 ** (self.iters + 1)
			numer = self.m_t[task][name] / bias_corr1
		else:
			numer = dw
		if self.optim=='rms' or self.optim=='adam':
			self.v_t[task][name] = self.beta2 * self.v_t[task][name] + (1 - self.beta2) * dw ** 2
		else:
			self.v_t[task][name] = self.v_t[task][name] + dw ** 2
		if task>0:
			alpha_add = []
			for t in range(task):
				corr = torch.dot(self.m_t[t][name].reshape(-1) / torch.norm(self.m_t[t][name]), self.m_t[task][name].reshape(-1) / torch.norm(self.m_t[task][name]))
				alpha_add += [(-corr).cpu().numpy()]
			alpha_add += [-1.]
			alpha_add = torch.from_numpy(np.array(alpha_add)).to(DEVICE)
			alpha_add_ = torch.exp(self.b*alpha_add).float()
			self.alpha_add_[task][name] = alpha_add_.cpu().numpy()
		else:
			alpha_add_ = torch.from_numpy(np.array([1.0])).to(DEVICE)

		for t in range(task):
			new_v = self.v_t[t][name].unsqueeze(0) if t==0 else torch.cat((new_v, self.v_t[t][name].unsqueeze(0)), dim=0)
		new_v = self.v_t[task][name].unsqueeze(0) if new_v is None else torch.cat((new_v, self.v_t[task][name].unsqueeze(0)), dim=0)
		denom = (torch.sqrt(torch.einsum(eq, alpha_add_.float(), new_v))/ math.sqrt(bias_corr2)) + 1e-8
		return - (lr * numer / denom)

	def step(self, model, task, iters, lr=None):
		self.iters = iters
		state_dict = model.state_dict()
		for i, (name, param) in enumerate(state_dict.items()):
			if name.split('.')[-1] in ['running_mean', 'num_batches_tracked', 'running_var']:
				continue
			for n, v in model.named_parameters():
				if n == name:
					break
			update = self.manual_update(name, v.grad, task, self.lr if lr is None else lr)
			state_dict[name].data.copy_(param + update.reshape(param.shape))
		return state_dict


def store_alpha(optimizer, task_id, iter, alpha_mean):
	for j in optimizer.alpha_add_[task_id - 1]:
		alphas = optimizer.alpha_add_[task_id - 1][j]
		if iter==0:
			alpha_mean[j] = alphas
		else:
			alpha_mean[j] = (alpha_mean[j]*iter + alphas)/(iter+1)
	return alpha_mean
