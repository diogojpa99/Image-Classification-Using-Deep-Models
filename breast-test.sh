dataset_n="CMMD-only_mass"

datapath="../../data/CMMD/$dataset_n"
testset_name="$dataset_n"
dataset_name="DDSM+CBIS-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('vgg16')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He_No_Resize"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="xFinetuned_Models/Binary/Benign_vs_Malignant/DDSM+CBIS-Benign_vs_Malignant/Baselines-Finetune-Breast-Benign_vs_Malignant-DDSM+CBIS-Benign_vs_Malignant-vgg16-loader_Gray_PIL_Loader_Wo_He_No_Resize-Date_20240316_180821/Baseline-vgg16-best_checkpoint.pth"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-$base-loader_$loader-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --input_size 23 \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_loader $loader \
    --breast_clahe \
    --clahe_clip_limit 5.0 \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "xTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

dataset_n="MIAS_CLAHE-Benign_vs_Malignant"

datapath="../Data/$dataset_n"
testset_name="$dataset_n"
dataset_name="DDSM+CBIS-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('vgg16')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He_No_Resize"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="xFinetuned_Models/Binary/Benign_vs_Malignant/DDSM+CBIS-Benign_vs_Malignant/Baselines-Finetune-Breast-Benign_vs_Malignant-DDSM+CBIS-Benign_vs_Malignant-vgg16-loader_Gray_PIL_Loader_Wo_He_No_Resize-Date_20240316_180821/Baseline-vgg16-best_checkpoint.pth"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-$base-loader_$loader-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --input_size 23 \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_loader $loader \
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "xTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

exit 0 

###########################################################################################################


dataset_n="DDSM+CBIS-Benign_vs_Malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../Data/$dataset_n"
testset_name="$dataset_n"
dataset_name="$dataset_n"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "wFinetuned_Models/Binary/$classification_problem/$dataset_name/antialias/" -type d -path "*/Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-loader_$loader-antialias_on-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"
    echo "$ckpt_path"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-loader_$loader-antialias_on-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_loader $loader \
    --breast_antialias \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "wTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

################################################################################################################################################################

dataset_n="CMMD-only_mass-processed_crop_CLAHE"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$dataset_n"
testset_name="$dataset_n"
dataset_name="DDSM+CBIS-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "wFinetuned_Models/Binary/$classification_problem/$dataset_name/antialias/" -type d -path "*/Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-loader_$loader-antialias_on-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"
    echo "$ckpt_path"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-loader_$loader-antialias_on-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_loader $loader \
    --breast_antialias \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "wTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done


################################################################################################################################################################

dataset_n="MIAS_CLAHE-Benign_vs_Malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../Data/$dataset_n"
testset_name="$dataset_n"
dataset_name="DDSM+CBIS-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "wFinetuned_Models/Binary/$classification_problem/$dataset_name/antialias/" -type d -path "*/Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-loader_$loader-antialias_on-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"
    echo "$ckpt_path"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-loader_$loader-antialias_on-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_antialias \
    --breast_loader $loader \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "wTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

exit 0

################################################################################################################################################################


dataset_n="DDSM+CBIS-Benign_vs_Malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../Data/$dataset_n"
testset_name="$dataset_n"
dataset_name="$dataset_n"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "wFinetuned_Models/Binary/$classification_problem/$dataset_name/" -type d -path "*/Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-loader_$loader-antialias_on-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"
    echo "$ckpt_path"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-loader_$loader-antialias_off-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_loader $loader \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "wTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

################################################################################################################################################################

dataset_n="CMMD-only_mass-processed_crop_CLAHE"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$dataset_n"
testset_name="$dataset_n"
dataset_name="DDSM+CBIS-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "wFinetuned_Models/Binary/$classification_problem/$dataset_name/" -type d -path "*/Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-loader_$loader-antialias_on-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"
    echo "$ckpt_path"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-loader_$loader-antialias_on-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_loader $loader \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "wTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done


################################################################################################################################################################

dataset_n="MIAS_CLAHE-Benign_vs_Malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../Data/$dataset_n"
testset_name="$dataset_n"
dataset_name="DDSM+CBIS-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "wFinetuned_Models/Binary/$classification_problem/$dataset_name/" -type d -path "*/Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-loader_$loader-antialias_on-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"
    echo "$ckpt_path"
    logdir="Baselines-Test-$dataset_type-$classification_problem-Test_set_$testset_name-Train_set_$dataset_name-loader_$loader-antialias_on-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python main.py \
    --eval \
    --resume $ckpt_path \
    --model $base \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 8 \
    --class_weights "balanced" \
    --breast_loader $loader \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "wTests/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

exit 0