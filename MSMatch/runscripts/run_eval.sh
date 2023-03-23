#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DEVICE=0                          
DATASET="thraws_swir_test"  #Dataset to use: Options are eurosat_ms, eurosat_rgb
NET="efficientnet-lite0"
BATCH_SIZE=8
FIXMATCH_FOLDER="/data/PyDeepLearning/END2END/MSMatch/"
SEED=2
RED='\033[0;31m'
BLACK='\033[0m'
CSV_PATH="."
CONFUSION_MATRIX="--export_confusion_matrix --csv_path $CSV_PATH --plot_confusion_matrix"
#create save location


LOAD_PATH=r"C:\Users\meoni\Documents\ESA\Projects\END2END\MSMatch\checkpoint\thraws_swir_train\FixMatch_archefficientnet-lite0_batch8_confidence0.95_lr0.03_uratio4_wd0.00075_wu1.0_seed0_numlabels800_optSGD\model_best.pth"

#switch to fixmatch folder for execution
cd $FIXMATCH_FOLDER
echo -e "Using GPU ${RED} $CUDA_VISIBLE_DEVICES ${BLACK}."

#Remove "echo" to launch the script.
python eval.py --load_path $LOAD_PATH --net $NET --batch_size $BATCH_SIZE --dataset $DATASET --seed $SEED $CONFUSION_MATRIX 
