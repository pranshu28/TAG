# TAG
Task-based Accumulated Gradients for Lifelong learning.

In this work, While existing methods employ the general task-agnostic stochastic gradient descent update rule, we propose a task-aware optimizer that adapts the learning rate based on the relatedness among tasks. We utilize the directions taken by the parameters during the updates by accumulating the gradients specific to each task. These task-based accumulated gradients act as a knowledge base that is maintained and updated throughout the stream. We empirically show that our proposed adaptive learning rate not only accounts for catastrophic forgetting but also allows positive backward transfer. We also show that our method performs better than several state-of-the-art methods in lifelong learning on complex datasets with a large number of tasks. 



## 1. Code Structure
The high level structure of the code is as follows:

```
root
├── stable_sgd
│   
└── external_libs
    └── continual_learning_algorithms
    └── hessian_eigenthings
```

1. `stable_sgd`   : implementations of our stable and plastic training regimen for SGD (in Pytorch).      
2. `external_libs`: third-party implementations we used for our experiments such as:   
    2.1 `continual_learning_algorithms` Open source implementations for A-GEM, ER-Reservoir, and EWC (in Tensorflow).   
    2.2 `hessian_eigenthings`: Open source implementation of deflated power iteration for eigenspectrum calculations (in Pytorch).  

## 2. Setup & Installation
The code is tested on Python 3.6+, PyTorch 1.5.0, and Tensorflow 1.15.2. In addition, there are some other numerical and visualization libraries that are included in ``requirements.txt`` file. However, for convenience, we provide a script for setup:   
```
bash setup_and_install.sh
```

## 3. Replicating the Results
We provide scripts to replicate the results:   
 * 3.1 Run ```bash replicate_experiment_1.sh``` for experiment 1 (stable vs plastic).   
 * 3.2 Run ```bash replicate_experiment_2.sh``` for experiment 2 (Comparison with other methods with 20 tasks).
 * 3.3 Run ```bash replicate_appendix_c5.sh```  for the experiment in appendix C5 (Stabilizing other methods).
 
For faster replication, here we have only 3 runs per method per experiment, but we used 5 runs for the reported results.