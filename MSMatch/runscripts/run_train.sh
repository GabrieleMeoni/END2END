#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DEVICE=0
FIXMATCH_FOLDER="/data/PyDeepLearning/END2END/MSMatch/"
SAVE_LOCATION="/data/PyDeepLearning/END2END/MSMatch/checkpoints/" #Where tensorboard output will be written
SAVE_NAME="iter2"                             
DATASET="thraws_swir"   #Dataset to use: Options are eurosat_ms, eurosat_rgb, aid, ucm
NET=efficientnet-lite0 #Options are wideResNet,efficientnet-lite0, efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5,...  
UNLABELED_RATIO=4
BATCH_SIZE=8
N_EPOCH=500                    #Set NUM_TRAIN_ITER = N_EPOCH * NUM_EVAL_ITER * 32 / BATCH_SIZE
NUM_EVAL_ITER=1000            #Number of iterations 
NUM_TRAIN_ITER=$(($N_EPOCH * $NUM_EVAL_ITER * 32/ $BATCH_SIZE))
SEED=0
WEIGHT_DECAY=0.00075
LR=0.03
RED='\033[0;31m'
BLACK='\033[0m'
URL_DIST="tcp://127.0.0.1:10007" #change port to avoid conflicts to allow multiple multi-gpu runs
USE_MCC_FOR_BEST="--use_mcc_for_best" # Leave empty to select best model depending on accuracy. Use use_mcc_for_best to select it depending on the mcc metric.
#create save location
mkdir -p $SAVE_LOCATION

#Upsampling values.
TRAIN_UPS_EVENT=5
TRAIN_UPS_NOTEVENT=1
EVAL_UPS_EVENT=1
EVAL_UPS_NOTEVENT=1

NUM_LABELS_USED="600"

#switch to fixmatch folder for execution
cd $FIXMATCH_FOLDER
echo -e "Using GPU ${RED} $CUDA_VISIBLE_DEVICES ${BLACK}."

if [[ ${#CUDA_VISIBLE_DEVICES} > 1 ]]
then
	echo -e "${RED} Multi-GPU mode ${BLACK}"
	for NUM_LABELS in $NUM_LABELS_USED; do #Note: they are the total number of labels, not per class.
		#Remove "echo" to launch the script.
		python train.py --weight_decay $WEIGHT_DECAY --world-size 1 --rank 0 --multiprocessing-distributed --dist-url $URL_DIST --lr $LR --batch_size $BATCH_SIZE --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --num_labels $NUM_LABELS --save_name $SAVE_NAME --save_dir $SAVE_LOCATION --dataset $DATASET --net $NET --seed $SEED --uratio $UNLABELED_RATIO --train_upsample_event $TRAIN_UPS_EVENT --train_upsample_notevent $TRAIN_UPS_NOTEVENT --eval_upsample_event $EVAL_UPS_EVENT --eval_upsample_notevent $EVAL_UPS_NOTEVENT --overwrite $USE_MCC_FOR_BEST
		wait
	done
else
	for NUM_LABELS in $NUM_LABELS_USED; do #Note: they are the total number of labels, not per class.
		#Remove "echo" to launch the script.
		python train.py --weight_decay $WEIGHT_DECAY --rank 0 --gpu $DEVICE --lr $LR --batch_size $BATCH_SIZE --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --num_labels $NUM_LABELS --save_name $SAVE_NAME --save_dir $SAVE_LOCATION --dataset $DATASET --net $NET --seed $SEED --uratio $UNLABELED_RATIO --train_upsample_event $TRAIN_UPS_EVENT --train_upsample_notevent $TRAIN_UPS_NOTEVENT --eval_upsample_event $EVAL_UPS_EVENT --eval_upsample_notevent $EVAL_UPS_NOTEVENT --overwrite $USE_MCC_FOR_BEST
		wait
	done
fi
