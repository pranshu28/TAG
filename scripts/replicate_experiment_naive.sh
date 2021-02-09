#!/usr/bin/env bash

if [ $1 = "cifar" ]; then
    echo "************************ replicating experiment (Split CIFAR-100) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'

elif [ $1 = "imagenet" ]; then
    echo "************************ replicating experiment (Split-miniImageNet) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'


elif [ $1 = "cub" ]; then
    echo "************************ replicating experiment (Split-CUB) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.000025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

elif [ $1 = "5data" ]; then
    echo "************************ replicating experiment (5-dataset) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 5 --lr 0.0005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 7 --tag-opt 'rms'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'adam'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 7 --tag-opt 'adagrad'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 7 --tag-opt 'adam'

fi