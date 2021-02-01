from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


class EWC(object):
	def __init__(self, model, criterion):

		self.model = model
		self.criterion = criterion

		self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
		self._means = {}
		self.precision_matrices = {}
		for n, p in deepcopy(self.params).items():
			p.data.zero_()
			self.precision_matrices[n] = Variable(p.data).cuda()
		for n, p in deepcopy(self.params).items():
			self._means[n] = Variable(p.data).cuda()

	def update(self, model, t, loader):
		self.model = model
		self.model.eval()
		for n, p in deepcopy(self.params).items():
			self._means[n] = Variable(p.data).cuda()
		for x, y in loader:
			self.model.zero_grad()
			x = Variable(x).cuda()
			output = self.model(x, t + 1)
			loss = self.criterion(output, y.cuda())
			loss.backward()
			for n, p in self.model.named_parameters():
				self.precision_matrices[n].data = (self.precision_matrices[n].data + (p.grad.data ** 2) * t) / (t+1)
			break

	def penalty(self, model: nn.Module):
		loss = 0
		for n, p in model.named_parameters():
			_loss = self.precision_matrices[n] * (p - self._means[n]) ** 2
			loss += _loss.sum()
		return loss


class EWC_(object):
	def __init__(self, model, dataset, criterion):

		self.model = model
		self.dataset = dataset
		self.criterion = criterion

		self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
		self._means = {}
		self._precision_matrices = self._diag_fisher()

		for n, p in deepcopy(self.params).items():
			self._means[n] = Variable(p.data).cuda()

	def _diag_fisher(self):
		precision_matrices = {}
		for n, p in deepcopy(self.params).items():
			p.data.zero_()
			precision_matrices[n] = Variable(p.data).cuda()

		self.model.eval()
		for input in self.dataset:
			self.model.zero_grad()
			input = Variable(input.unsqueeze(0)).cuda()
			output = self.model(input).view(1, -1)
			label = output.max(1)[1].view(-1)
			loss = self.criterion(output, label)
			loss.backward()

			for n, p in self.model.named_parameters():
				precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

		precision_matrices = {n: p for n, p in precision_matrices.items()}
		return precision_matrices

	def penalty(self, model: nn.Module):
		loss = 0
		for n, p in model.named_parameters():
			_loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
			loss += _loss.sum()
		return loss
