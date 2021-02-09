from utils import *

import torch
from tqdm.auto import tqdm
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.utils.data as data

from collections import defaultdict


def orthonormalize(vectors, normalize=True, start_idx=0):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    if start_idx == 0 :
        start_idx = 1
    for i in range(start_idx, vectors.size(1)):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors


def project_vec(vec, proj_basis):
    if proj_basis.shape[1] > 0:  # param x basis_size
        dots = torch.matmul(vec, proj_basis)  # basis_size
        out = torch.matmul(proj_basis, dots.T)
        return out
    else:
        return torch.zeros_like(vec)


def parameters_to_grad_vector(parameters):
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        vec.append(param.grad.view(-1))
    return torch.cat(vec)


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


def get_n_trainable(model):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable


class Storage(data.Dataset):
    """
    A dataset wrapper used as a memory to store the data
    """
    def __init__(self):
        super(Storage, self).__init__()
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        return self.storage[index]

    def append(self,x):
        self.storage.append(x)

    def extend(self,x):
        self.storage.extend(x)


class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]

    def get_tensor(self):
        storage = [x.unsqueeze(-1) for x in self.storage]
        return torch.cat(storage, axis=1)


class OGD():
    def __init__(self, args, model, optimizer):
        super().__init__()
        self.criterion_fn = nn.CrossEntropyLoss()
        self.args = args

        self.model = model
        self.optimizer = optimizer

        # if self.config.is_split_cub :
        #     n_params = get_n_trainable(self.model)
        # elif self.config.is_split :
        #     n_params = count_parameter(self.model.linear)
        # else :
        n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]))
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).cuda())
        self.task_count = 0
        self.task_memory = None
        self.task_grad_memory = {}
        # self.mem_loaders = list()

    def _get_new_ogd_basis(self, train_loader, last=False):
        return self._get_neural_tangents(train_loader,
                                         optimizer=self.optimizer,
                                         model=self.model, last=last)

    def _get_neural_tangents(self, train_loader, optimizer, model, last):
        new_basis = []

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = self.to_device(inputs)
            targets = self.to_device(targets)

            out = model(inputs, None)
            out = apply_mask(targets, out, model.n_classes)

            label = targets.item()
            pred = out[0, label]

            optimizer.zero_grad()
            pred.backward()

            grad_vec = parameters_to_grad_vector(model.parameters())
            new_basis.append(grad_vec)
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor

    def to_device(self, tensor):
        if DEVICE=='cuda':
            return tensor.cuda()
        else :
            return tensor

    def optimizer_step(self, optimizer):
        cur_param = parameters_to_vector(self.model.parameters())
        grad_vec = parameters_to_grad_vector(self.model.parameters())
        # if self.do_ogd:
        proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis)
        new_grad_vec = grad_vec - proj_grad_vec
        # else:
        #     new_grad_vec = grad_vec
        cur_param -= self.args.lr * new_grad_vec
        vector_to_parameters(cur_param, self.model.parameters())

        # if self.config.is_split :
        #     task_key = str(self.task_id)
        #     # Update the parameters of the last layer without projection, when there are multiple heads)
        #     cur_param = parameters_to_vector(self.get_params_dict(last=True, task_key=task_key))
        #     grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True, task_key=task_key))
        #     cur_param -= lr * grad_vec
        #     vector_to_parameters(cur_param, self.get_params_dict(last=True, task_key=task_key))
        optimizer.zero_grad()
        return self.model

    def _update_mem(self, task_id, data_train_loader):
        # 2.Randomly decide the images to stay in the memory

        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.args.mem_size

        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(data_train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory.append(data_train_loader.dataset[ind])

        ####################################### Grads MEM ###########################

        # (e) Get the new non-orthonormal gradients basis
        # if self.do_ogd:
        ogd_train_loader = torch.utils.data.DataLoader(self.task_memory, batch_size=1, shuffle=False, num_workers=4)
        # Non orthonormalised basis
        new_basis_tensor = self._get_new_ogd_basis(ogd_train_loader)
        # print("new_basis_tensor shape",new_basis_tensor.shape)

        # (f) Ortonormalise the whole memorized basis
        # if self.config.is_split:
        #     n_params = count_parameter(self.model.linear)
        # else:
        n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        self.ogd_basis = self.to_device(self.ogd_basis)

        # if self.do_ogd :
        for t, mem in self.task_grad_memory.items():
            # Concatenate all data in each task
            task_ogd_basis_tensor = self.to_device(mem.get_tensor())
            self.ogd_basis = torch.cat([self.ogd_basis, task_ogd_basis_tensor], axis=1)
        self.ogd_basis = torch.cat([self.ogd_basis, new_basis_tensor], axis=1)
        # start_idx = (task_id - 1) * num_sample_per_task
        # print("the start idx of orthonormalisation if {start_idx}")
        self.ogd_basis = orthonormalize(self.ogd_basis, normalize=True)

        # (g) Store in the new basis
        ptr = 0
        # for t, mem in self.task_memory.items():
        task_mem_size = len(self.task_memory)

        # idxs_list = [i + ptr for i in range(task_mem_size)]
        # if self.config.gpu:
        # self.ogd_basis_ids[t] = torch.LongTensor(idxs_list).cuda()
        # else:
        #     self.ogd_basis_ids[t] = torch.LongTensor(idxs_list)

        self.task_grad_memory[task_id] = Memory()  # Initialize the memory slot
        for ind in range(task_mem_size):  # save it to the memory
            self.task_grad_memory[task_id].append(self.ogd_basis[:, ptr].cpu())
            ptr += 1
        # print("Used memory",ptr,"/",self.args.mem_size)

        # if self.do_ogd or self.do_ogd_plus :
        # loader = torch.utils.data.DataLoader(self.task_memory[task_id], batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        # self.mem_loaders.append(loader)
