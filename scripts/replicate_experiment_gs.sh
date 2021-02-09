#!/usr/bin/env bash

if [ $1 = "cifar" ]; then
    echo "************************ replicating experiment (Split CIFAR-100) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'ewc' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'stable' --gamma 0.9 --batch-size 10 --dropout 0.1 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --tag-opt 'rms'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset cifar100 --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'


elif [ $1 = "imagenet" ]; then
    echo "************************ replicating experiment (mini-imagenet) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'ewc' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'stable' --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset imagenet --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'


elif [ $1 = "cub" ]; then
    echo "************************ replicating experiment (cub) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'ewc' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 10

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'stable' --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset cub --tasks 20 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

elif [ $1 = "5data" ]; then
    echo "************************ replicating experiment (5data) ***********************"

    echo " >>>>>>>> Naive SGD "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5

    echo " >>>>>>>> EWC "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'ewc' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'ewc' --lambd 1

    echo " >>>>>>>> A-GEM "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1

    echo " >>>>>>>> ER "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1

    echo " >>>>>>>> Stable SGD "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'stable' --gamma 0.7 --batch-size 64 --dropout 0.0 --runs 5

    echo " >>>>>>>> Naive RMSProp "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'rms'

    echo " >>>>>>>> Naive Adagrad "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'adagrad'

    echo " >>>>>>>> Naive Adam "
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'lr' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'adam'

    echo " >>>>>>>> TAG-Adagrad"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adagrad'

    echo " >>>>>>>> TAG-Adam"
    python3 -m main --dataset 5data --tasks 5 --epochs-per-task 1 --hyp-gs 'tag' --gamma 1.0 --batch-size 64 --dropout 0.0 --runs 5 --opt 'tag' --b 5 --tag-opt 'adam'


fi