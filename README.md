# TAG: Task-based Accumulated Gradients for Lifelong learning.

In this work, we propose a task-aware optimizer called **TAG** that adapts the learning rate based on the relatedness among tasks. 
We utilize the directions taken by the parameters during the updates by accumulating the gradients specific to each task. 
These task-based accumulated gradients act as a knowledge base that is maintained and updated throughout the stream. 
In the experiments, we show that TAG not only accounts for catastrophic forgetting but also exhibits positive backward transfer. 
We also show that our method performs better than several state-of-the-art methods in lifelong learning on Split-CIFAR100, Split-miniImageNet, Split-CUB and 5-dataset. 
The overall implementation is adopted from the repository [imirzadeh/stable-continual-learning](https://github.com/imirzadeh/stable-continual-learning).

## Project Structure
The high level structure of the code is as follows:

```
root
├── main.py
├── models.py
├── tag_update.py
├── utils.py
│
├── data
│   └── data_loader.py
│   └── data_utils.py
│
├── existing_methods
│   └── agem.py
│   └── er.py
│   └── ewc.py
│
└── scripts
    └── replicate_experiment_baselines.sh
    └── replicate_experiment_gs.sh
    └── replicate_experiment_hybrid.sh
    └── replicate_experiment_naive.sh
    └── replicate_experiment_replay.sh

```

1. `main.py`   : Contains main function that imports datasets and models based on given arguments. It also contains implementation of Naive-optimizers and Stable SGD.      
2. `models.py`   : Implementation of different deep learning models used in this work.      
3. `tag_update.py`   : Implementation of our proposed parameter update method - TAG.      
4. `utils.py`   : Contains functions for setting seed, initializing experiments and logging etc.      
5. `data`: Code for downloading nad importing the datasets used in this work:
    
    5.1 `data_loader.py`:  Code for getting the data loaders ready for the given dataset and number of tasks.  
    5.2 `data_utils.py`:  Code for importing the datasets:
    * Mini-Imangenet and CUB datasets (borrowed from [optimass/Maximally_Interfered_Retrieval](https://github.com/optimass/Maximally_Interfered_Retrieval)).  
    * CIFAR100 (borrowed from [imirzadeh/stable-continual-learning](https://github.com/imirzadeh/stable-continual-learning)).
    * 5-dataset.
6. `existing_methods`: Implementations of the existing baselines used for our experiments:   
    
    6.1 `agem.py` A substantial part of implementation of A-GEM comes from the official GEM repository [facebookresearch/GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory).   
    6.2 `er.py`: Implementation of ER comes from the open source repository [optimass/Maximally_Interfered_Retrieval](https://github.com/optimass/Maximally_Interfered_Retrieval).  
    6.3 `ewc.py`: Implementation of EWC comes from the open source repository [moskomule/ewc.pytorch](https://github.com/moskomule/ewc.pytorch).  
7. `scripts`: Bash scripts for replicating the results shown in the paper.
 ___
 
## Setup & Installation
The code is tested on Python 3.6+ and PyTorch 1.5.0. We provide ``requirements.txt`` that contains other important packages and the command to install them is given below.
```
pip install -r requirements.txt
```
 ___

## Sources of the datasets
We download the datasets from the below links and store them in the `data` folder:
 * CIFAR100: torchvision
 * mini-Imangenet: [link](https://www.kaggle.com/whitemoon/miniimagenet)
 * CUB: Parse this [tar file](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view).
 * 5-dataset: It consists of the following datasets:
    * CIFAR10: torchvision
    * MNIST: torchvision
    * SVHN: torchvision
    * notMNIST: Parse the `notMNIST_small.tar` file that is downloaded from [this link](https://yaroslavvb.com/upload/notMNIST/).
    * FashionMNIST: torchvision

 ___

## Replicating the Results
We provide the following scripts to replicate the results:   
 * For experiment in the Section 4.1 (Naive-optimizers vs TAG-optimizers), run: 
 ```
 bash scripts/replicate_experiment_naive.sh <dataset>
 ``` 
 * For experiment in the Section 4.2 (Comparison with other baselines), run:
  ```
  bash scripts/replicate_experiment_baselines.sh <dataset> 1
  ``` 
 * For experiment in the Section 4.3 (Combining TAG with other baselines), run: 
 ```
 bash scripts/replicate_appendix_hybrid.sh <dataset>
 ```
 
For replicating the results given in the Appendix:
 * Comparison with other baselines by training on 5 epochs per task: 
 ```
 bash scripts/replicate_experiment_baselines.sh <dataset> 5
 ```
 * Comparing TAG results with A-GEM and ER having bigger memory sizes:
 ```
 bash scripts/replicate_appendix_replay.sh <dataset>
 ```

We run the following command to replicate the grid-search performed to choose the best hyper-parameters set:
 ```
 bash scripts/replicate_appendix_gs.sh <dataset>
 ```

In all above cases, `<dataset>` can be one of the following:
 * `cifar`: Split-CIFAR100
 * `mini_imagenet`: Split-miniImageNet
 * `cub`: Split-CUB
 * `5data`: 5-dataset
 
 ___
