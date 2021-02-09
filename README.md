# TAG: Task-based Accumulated Gradients for Lifelong learning.

In this work, we propose a task-aware optimizer called **TAG** that adapts the learning rate based on the relatedness among tasks. 
We utilize the directions taken by the parameters during the updates by accumulating the gradients specific to each task. 
These task-based accumulated gradients act as a knowledge base that is maintained and updated throughout the stream. 
In the experiments, we show that TAG not only accounts for catastrophic forgetting but also allows positive backward transfer. 
We also show that our method performs better than several state-of-the-art methods in lifelong learning on Split-CIFA100, Split-miniImageNet, Split-CUB and 5-dataset. 
The overall implementation is based on the repository [imirzadeh / stable-continual-learning](https://github.com/imirzadeh/stable-continual-learning).

## Project Structure
The high level structure of the code is as follows:

```
root
├── main.py
├── models.py
├── utils.py
├── tag_update.py
│
├── existing_methods
│   └── agem.py
│   └── er.py
│   └── ewc.py
│   └── ogd.py
│
└── data
    └── data_utils.py
    └── data_utils_2.py   
```

1. `main.py`   : Contains main function that imports datasets and models based on given arguments. It also contains implementation of Naive-optimizers and Stable SGD.      
2. `models.py`   : Implementation of different deep learning models used in this work.      
3. `tag_update.py`   : Implementation of our proposed parameter update method - **TAG**.      
4. `data`: Code for downloading nad importing the datasets used in this work:
    
    4.1 `data_utils.py`:  Code for importing CIFAR100 (similar as [imirzadeh / stable-continual-learning](https://github.com/imirzadeh/stable-continual-learning)) and 5-dataset.  
    4.2 `data_utils_2.py`:  Code for importing mini-Imangenet and CUB datasets (most part of the implementation comes from [optimass / Maximally_Interfered_Retrieval](https://github.com/optimass/Maximally_Interfered_Retrieval)).  
5. `existing_methods`: Implementations of the existing baselines used for our experiments:   
    
    5.1 `agem.py` Implementations for A-GEM.   
    5.2 `er.py`: Implementation comes from the open source implementation of ER-Reservoir in the repository [optimass / Maximally_Interfered_Retrieval](https://github.com/optimass/Maximally_Interfered_Retrieval).  
    5.3 `ewc.py`: Open source implementation of EWC.  
    5.4 `ogd.py`: A substantial part of this implementation of OGD comes from the repository [MehdiAbbanaBennani / continual-learning-ogdplus](https://github.com/MehdiAbbanaBennani/continual-learning-ogdplus). 

## 2. Setup & Installation
The code is tested on Python 3.6+ and PyTorch 1.5.0. We also provide ``requirements.txt`` that contains other important packages and the command to install them is given below.
```
bash setup_and_install.sh
```

## 3. Replicating the Results
We provide the following scripts to replicate the results:   
 * 3.1 Run ```bash replicate_experiment_1.sh``` for experiment 1 (stable vs plastic).   
 * 3.2 Run ```bash replicate_experiment_2.sh``` for experiment 2 (Comparison with other methods with 20 tasks).
 * 3.3 Run ```bash replicate_appendix_c5.sh```  for the experiment in appendix C5 (Stabilizing other methods).
 
For faster replication, here we have only 3 runs per method per experiment, but we used 5 runs for the reported results.