#!/usr/bin/env bash

cd ~/anaconda3/bin
source activate gcn_env
cd ~/pranshu/research/mtl_ll/tag

if [ $1 = "cifar10" ]; then
    echo "************************ replicating experiment 2 (Split CIFAR-100) ***********************"

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --multi 1
##
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.1 --gamma 0.9 --batch-size 64 --dropout 0.0 --runs 3

#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --opt 'agem' --mem-size 20
#
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --opt 'er' --mem-size 20
#
#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'rms'

#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'adagrad'

#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'adam'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset cifar100 --tasks 10 --epochs-per-task 200 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'adam'


elif [ $1 = "imagenet" ]; then
    echo "************************ replicating experiment 2 (mini-imagenet) ***********************"

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --multi 1
##
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.05 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 3
#
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'agem' --mem-size 5

#    echo " >>>>>>>> ER "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'er' --mem-size 1

#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'rms'
#
    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'adagrad'

#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'adam'

#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 10 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'adam'

elif [ $1 = "5data" ]; then
    echo "************************ replicating experiment 2 (5data) ***********************"

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 100 --lr 0.01 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 100 --lr 0.01 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --multi 1
#
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 100 --lr 0.005 --gamma 0.9 --batch-size 64 --dropout 0.0 --runs 3

#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 100 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --opt 'agem' --mem-size 60

    echo " >>>>>>>> ER "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 100 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --opt 'er' --mem-size 60

#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 100 --lr 0.0001 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 100 --lr 0.00025 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'rms'

#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset 5data --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'adagrad'
#
#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset 5data --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'param' --b 5 --tag-opt 'adam'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset 5data --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset 5data --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --opt 'adam'


fi