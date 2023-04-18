#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DEVICE=0
FIXMATCH_FOLDER="/home/roberto/PythonProjects/END2END/MSMatch/"
SAVE_LOCATION="/home/roberto/PythonProjects/END2END/MSMatch/checkpoints/" # Where tensorboard output will be written
DATASET="thraws_swir_train"   #Dataset to use: Options are eurosat_ms, eurosat_rgb, aid, ucm
TEST_DATASET="thraws_swir_test"  
NET=efficientnet-lite0 #Options are wideResNet,efficientnet-lite0, efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5,...  
UNLABELED_RATIO=4 #Unlabeled ratio. 4 means 4 times more unlabeled than labeled data. 
BATCH_SIZE=18
EVAL_SPLIT_RATIO=0.1 #Evaluation split percentage over the whole train/eval dataset.
N_EPOCH=50                    #Set NUM_TRAIN_ITER = N_EPOCH * NUM_EVAL_ITER * 32 / BATCH_SIZE
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
TRAIN_UPS_EVENT=1
TRAIN_UPS_NOTEVENT=1
EVAL_UPS_EVENT=1
EVAL_UPS_NOTEVENT=1
EVAL_BATCH_SIZE=32

NUM_LABELS_USED="600"

# loop for TRAIN_UPS_EVENT values in the 3,5,7,9:
for TRAIN_UPS_EVENT in 3 5 7 9; do
	SAVE_NAME="esr_${EVAL_SPLIT_RATIO}_Ups_t${TRAIN_UPS_EVENT}_n${TRAIN_UPS_NOTEVENT}_e${EVAL_UPS_EVENT}_ne${EVAL_UPS_NOTEVENT}_b${EVAL_BATCH_SIZE}_l${NUM_LABELS_USED}_u${UNLABELED_RATIO}"
	#switch to fixmatch folder for execution
	cd $FIXMATCH_FOLDER
	echo -e "Using GPU ${RED} $CUDA_VISIBLE_DEVICES ${BLACK}."

	if [[ ${#CUDA_VISIBLE_DEVICES} > 1 ]]
	then
		echo -e "${RED} Multi-GPU mode ${BLACK}"
		for NUM_LABELS in $NUM_LABELS_USED; do #Note: they are the total number of labels, not per class.
			#Remove "echo" to launch the script.
			python train.py --weight_decay $WEIGHT_DECAY --world-size 1 --rank 0 --multiprocessing-distributed --dist-url $URL_DIST --lr $LR --batch_size $BATCH_SIZE --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --num_labels $NUM_LABELS --save_name $SAVE_NAME --save_dir $SAVE_LOCATION --dataset $DATASET --net $NET --seed $SEED --uratio $UNLABELED_RATIO --train_upsample_event $TRAIN_UPS_EVENT --train_upsample_notevent $TRAIN_UPS_NOTEVENT --eval_upsample_event $EVAL_UPS_EVENT --eval_upsample_notevent $EVAL_UPS_NOTEVENT --overwrite $USE_MCC_FOR_BEST --test_dataset $TEST_DATASET --eval_split_ratio $EVAL_SPLIT_RATIO --eval_batch_size $EVAL_BATCH_SIZE
			wait
		done
	else
		for NUM_LABELS in $NUM_LABELS_USED; do #Note: they are the total number of labels, not per class.
			#Remove "echo" to launch the script.
			python train.py --weight_decay $WEIGHT_DECAY --rank 0 --gpu $DEVICE --lr $LR --batch_size $BATCH_SIZE --num_train_iter $NUM_TRAIN_ITER --num_eval_iter $NUM_EVAL_ITER --num_labels $NUM_LABELS --save_name $SAVE_NAME --save_dir $SAVE_LOCATION --dataset $DATASET --net $NET --seed $SEED --uratio $UNLABELED_RATIO --train_upsample_event $TRAIN_UPS_EVENT --train_upsample_notevent $TRAIN_UPS_NOTEVENT --eval_upsample_event $EVAL_UPS_EVENT --eval_upsample_notevent $EVAL_UPS_NOTEVENT --overwrite $USE_MCC_FOR_BEST --test_dataset $TEST_DATASET --eval_split_ratio $EVAL_SPLIT_RATIO --eval_batch_size $EVAL_BATCH_SIZE
			wait
		done
	fi
done
