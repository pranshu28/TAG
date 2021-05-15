#!/usr/bin/env bash

if [ $1 = "cifar" ]; then
    echo "************************ replicating experiment (Split CIFAR-100) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.1 --gamma 0.9 --batch-size 10 --dropout 0.1 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> Multi-task "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task $2 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1

elif [ $1 = "mini_imagenet" ]; then
    echo "************************ replicating experiment (Split-miniImageNet) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.05 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.05 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.05 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.1 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> Multi-task "
    python3 -m main --dataset mini_imagenet --tasks 20 --epochs-per-task $2 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1

elif [ $1 = "cub" ]; then
    echo "************************ replicating experiment (Split-CUB) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.05 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.000025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> Multi-task "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task $2 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1

elif [ $1 = "5data" ]; then
    echo "************************ replicating experiment (5-dataset) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 100

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.1 --gamma 0.7 --batch-size 64 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.0005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 7 --tag-opt 'rms'

    echo " >>>>>>>> Multi-task "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task $2 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --multi 1


fi
