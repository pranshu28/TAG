#!/usr/bin/env bash

cd ~/anaconda3/bin
source activate gcn_env
cd ~/pranshu/research/mtl_ll/tag

if [ $1 = "cifar" ]; then
    echo "************************ replicating experiment 2 (Split CIFAR-100) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1
#
    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5
##    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.15 --gamma 0.85 --batch-size 10 --dropout 0.1 --runs 5

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1
#
    echo " >>>>>>>> ER "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
##    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 10
#
    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'rms'

#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adagrad'

#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adam'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'


elif [ $1 = "imagenet" ]; then
    echo "************************ replicating experiment 2 (mini-imagenet) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1
#
    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.05 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5
#
    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
##    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 10
#
    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'
#
        echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adagrad'

#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adam'

#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

elif [ $1 = "cub" ]; then
    echo "************************ replicating experiment 2 (cub) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
##    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 10

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.000025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'rms'

#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adagrad'
#
#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adam'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'


fi