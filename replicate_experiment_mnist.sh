#!/usr/bin/env bash

cd ~/anaconda3/bin
source activate gcn_env
cd ~/pranshu/research/mtl_ll/tag

if [ $1 = "permute" ]; then
    echo "************************ replicating experiment (Permute MNIST) ***********************"

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
##
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.1 --gamma 0.55 --batch-size 10 --dropout 0.0 --runs 5
##
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1
#
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
##
#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.000075 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.000075 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'rms'

#    echo " >>>>>>>> Multi-task "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --multi 1

#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adagrad'

#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adam'
#
#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset perm-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'


elif [ $1 = "rotate" ]; then
    echo "************************ replicating experiment (Rotated MNIST) ***********************"

#    echo " >>>>>>>> Naive SGD "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5
##
#    echo " >>>>>>>> Stable SGD "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.01 --gamma 0.9 --batch-size 10 --dropout 0.0 --runs 5
##
#    echo " >>>>>>>> A-GEM "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'agem' --mem-size 1
#
#    echo " >>>>>>>> ER "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.1 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'er' --mem-size 1
#
#    echo " >>>>>>>> Naive RMSProp "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'rms'

    echo " >>>>>>>> TAG-RMSProp"
    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.00025 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'rms'
#
#    echo " >>>>>>>> Multi-task "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.01 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 3 --multi 1

#    echo " >>>>>>>> TAG-Adagrad"
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adagrad'

#    echo " >>>>>>>> TAG-Adam"
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'param' --b 5 --tag-opt 'adam'

#    echo " >>>>>>>> Naive Adagrad "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adagrad'

#    echo " >>>>>>>> Naive Adam "
#    python3 -m main --dataset rot-mnist --tasks 10 --epochs-per-task 1 --lr 0.005 --gamma 1.0 --batch-size 10 --dropout 0.0 --runs 5 --opt 'adam'

fi
