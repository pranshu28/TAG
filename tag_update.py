import math
from utils import *


class TAG(object):
	"""
	Implementation of our proposed TAG optimizer
	"""
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
		self.m_t_norms = {}
		for task in range(num_tasks):
			self.v_t[task] = {}
			self.m_t[task] = {}
			self.m_t_norms[task] = {}
			self.alpha_add_[task] = {}
			for (name, param) in model.named_parameters():
				if task == 0:
					self.v[name] = torch.zeros_like(param).to(args.device)
					self.m[name] = torch.zeros_like(param).to(args.device)
				self.alpha_add_[task][name] = np.array([1])
				self.v_t[task][name] = torch.zeros_like(param).to(args.device)
				self.m_t[task][name] = torch.zeros_like(param).to(args.device)
				self.m_t_norms[task][name] = torch.zeros_like(param).to(args.device)

	def zero_grad(self):
		return self.model.zero_grad()

	def update_all(self, task_id):
		"""
		Normalize the current task-based first moments (that will remain fixed)
		"""
		for name, v in self.model.named_parameters():
			self.m_t_norms[task_id][name] = self.m_t[task_id][name].reshape(-1) / torch.norm(self.m_t[task_id][name])

	def update_naive(self, param_name, param_grad):
		"""
		Use the naive-optimizer update
		:param param_name: Parameter identity
		:param param_grad: Gradient associated with the given parameter
		:return: New update to the given parameter
		"""
		if self.optim=='rms':
			self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * param_grad ** 2
		else:
			self.v[param_name] += param_grad ** 2
		denom = torch.sqrt(self.v[param_name]) + 1e-8
		return - (self.lr * param_grad / denom)

	def update_tag(self, param_name, param_grad, task_id):
		"""
		Update Task-based accumulated gradients, calculate alpha and return the new updates
		:param param_name: Parameter identity
		:param param_grad: Gradient associated with the given parameter
		:param task_id: Current task identity
		:return: New update to the given parameter
		"""
		bias_corr1, bias_corr2 = 1, 1
		eq = {1:'n,nh->h', 2:'n,nhw->hw', 3:'n,nhwc->hwc', 4: 'n,nhwvd->hwvd', 5:'n,nhwzxc->hwzxc'}[len(param_grad.shape)]
		new_v = None

		# Update task-based first moment
		self.m_t[task_id][param_name] = self.beta1 * self.m_t[task_id][param_name] + (1 - self.beta1) * param_grad

		# Change numerator based on the optimizer
		if self.optim=='adam':
			bias_corr1, bias_corr2 = 1 - self.beta1 ** (self.iters + 1), 1 - self.beta2 ** (self.iters + 1)
			numer = self.m_t[task_id][param_name] / bias_corr1
		else:
			numer = param_grad

		# Update task-based second moments based on the optimizer
		if self.optim=='rms' or self.optim=='adam':
			self.v_t[task_id][param_name] = self.beta2 * self.v_t[task_id][param_name] + (1 - self.beta2) * param_grad ** 2
		else:
			self.v_t[task_id][param_name] = self.v_t[task_id][param_name] + param_grad ** 2

		# Get new alphas by computing correlation using task-based first moments
		if task_id>0:
			alpha_add = []
			for t in range(task_id):
				corr = torch.dot(self.m_t[task_id][param_name].reshape(-1) / torch.norm(self.m_t[task_id][param_name]),
				                 self.m_t_norms[t][param_name])
				alpha_add += [(-corr).cpu().numpy()]
			alpha_add += [-1.]
			alpha_add = torch.from_numpy(np.array(alpha_add)).to(DEVICE)
			alpha_add_ = torch.exp(self.b*alpha_add).float()
		else:
			alpha_add_ = torch.from_numpy(np.array([1.0] * (task_id + 1))).to(DEVICE)
		self.alpha_add_[task_id][param_name] = alpha_add_.cpu().numpy()

		# Concatenate all task-based second moments and compute inner product with alphas
		for t in range(task_id):
			new_v = self.v_t[t][param_name].unsqueeze(0) \
					if t==0 \
					else torch.cat((new_v, self.v_t[t][param_name].unsqueeze(0)), dim=0)
		new_v = self.v_t[task_id][param_name].unsqueeze(0) \
				if new_v is None \
				else torch.cat((new_v, self.v_t[task_id][param_name].unsqueeze(0)), dim=0)
		denom = (torch.sqrt(torch.einsum(eq, alpha_add_.float(), new_v))/ math.sqrt(bias_corr2)) + 1e-8

		return - (self.lr * numer / denom)

	def step(self, model, task_id, step):
		"""
		Perform update over the parameters
		:param model: Current model
		:param task_id: Current task id (t)
		:param step: Current Step (n)
		:return:
		"""
		self.iters = step
		state_dict = model.state_dict()
		for i, (name, param) in enumerate(state_dict.items()):
			if name.split('.')[-1] in ['running_mean', 'num_batches_tracked', 'running_var']:
				continue
			for n, v in model.named_parameters():
				if n == name:
					break
			if v.grad is None:
				continue
			update = self.update_tag(name, v.grad, task_id)
			state_dict[name].data.copy_(param + update.reshape(param.shape))
		return state_dict


def store_alpha(tag_optimizer, task_id, iter, alpha_mean=None):
	"""
	Collects alpha values for given task (t) and current step (n)
	:param tag_optimizer: Object of the class tag_opt()
	:param task_id: Current task identity
	:param iter: Current step in the epoch
	:return: alpha_mean: Dictionary with previous task ids as keys
	"""
	for tau in tag_optimizer.alpha_add_[task_id]:
		alphas = tag_optimizer.alpha_add_[task_id][tau]
		if iter==0:
			alpha_mean[tau] = alphas
		else:
			alpha_mean[tau] = (alpha_mean[tau]*iter + alphas)/(iter+1)
	return alpha_mean
