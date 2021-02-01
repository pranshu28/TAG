#!/usr/bin/env bash

cd ~/anaconda3/bin
source activate gcn_env
cd ~/pranshu/research/mtl_ll/tag

if [ $1 = "cifar" ]; then
    echo "************************ replicating experiment (Split CIFAR-100) ***********************"
#
#    echo " >>>>>>>> Multi-task "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
#
#    echo " >>>>>>>> EWC "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'ewc' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 2 --opt 'ewc' --lambd 1

    echo " >>>>>>>> TAG-EWC "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-ewc' --lambd 1 --b 5 --tag-opt 'rms'
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --seed 2 --opt 'tag-ewc' --lambd 1 --b 5 --tag-opt 'rms'

#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 5
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 10
#
#    echo " >>>>>>>> TAG-A-GEM "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-agem' --mem-size 1 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 5
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 10
#
#    echo " >>>>>>>> TAG-ER "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-er' --mem-size 1 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 0.9 --batch-size 10 --dropout 0.1 --runs 5
#
#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'
##
#    echo " >>>>>>>> TAG-RMSProp"
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'
#
#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'
#
#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'
#
#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'

#    echo " >>>>>>>> OGD "
#    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 32 --dropout 0.0 --runs 5 --opt 'ogd' --mem-size 100


elif [ $1 = "imagenet" ]; then
    echo "************************ replicating experiment (Split-miniImageNet) ***********************"

#    echo " >>>>>>>> Multi-task "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
###
#    echo " >>>>>>>> EWC "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'ewc' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 2 --opt 'ewc' --lambd 1

    echo " >>>>>>>> TAG-EWC "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-ewc' --lambd 1 --b 5 --tag-opt 'rms'
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 2 --seed 3 --opt 'tag-ewc' --lambd 1 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 5
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 10
#
#    echo " >>>>>>>> TAG-A-GEM "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-agem' --mem-size 1 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 5
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 10
#
#    echo " >>>>>>>> TAG-ER "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-er' --mem-size 1 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.05 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5
#
#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'
#
#    echo " >>>>>>>> TAG-RMSProp"
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'
#
#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'
#
#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'

#    echo " >>>>>>>> OGD "
#    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ogd' --mem-size 100


elif [ $1 = "cub" ]; then
    echo "************************ replicating experiment (Split-CUB) ***********************"

#    echo " >>>>>>>> Multi-task "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --multi 1

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
##
    echo " >>>>>>>> EWC "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 10
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'ewc' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 2 --opt 'ewc' --lambd 10
#
    echo " >>>>>>>> TAG-EWC "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.000025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-ewc' --lambd 10 --b 5 --tag-opt 'rms'

#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 10
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 5
#
#    echo " >>>>>>>> TAG-A-GEM "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.000025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-agem' --mem-size 1 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 5
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 10
#
#    echo " >>>>>>>> TAG-ER "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.000025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag-er' --mem-size 1 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5
#
##
#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'
#
#    echo " >>>>>>>> TAG-RMSProp"
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.000025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'
##
#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'
#
#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'
#
#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.0001 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

###    echo " >>>>>>>> OGD "
###    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 1 --opt 'ogd' --mem-size 50

elif [ $1 = "5data" ]; then
    echo "************************ replicating experiment (5-dataset) ***********************"

#    echo " >>>>>>>> Multi-task "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --multi 1

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5
###
#    echo " >>>>>>>> EWC "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 100
#
    echo " >>>>>>>> TAG-EWC "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag-ewc' --lambd 100 --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 10
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 5
#
#    echo " >>>>>>>> TAG-A-GEM "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag-agem' --mem-size 1 --b 5 --tag-opt 'rms'
##
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'er' --mem-size 5
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.05 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'er' --mem-size 10
#
#    echo " >>>>>>>> TAG-ER "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag-er' --mem-size 1 --b 5 --tag-opt 'rms'
##
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.1 --gamma 0.7 --batch-size 64 --dropout 0.0 --runs 5
#
#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'rms'
#
#    echo " >>>>>>>> TAG-RMSProp"
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.0005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'adagrad'
#
#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'adam'
#
#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'
##
#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.001 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'

##    echo " >>>>>>>> OGD "
##    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'ogd' --mem-size 100
#


fi