###### General Settings ######
baselines=('efficientnet_b3' 'deit_small_patch16_224')

batch=128
n_classes=2
loader="Gray_PIL_Loader_Wo_He_No_Resize"

########################################### DDSM - Mass_ Normal #############################################################

trainset="DDSM_CLAHE-mass_normal"
testset="DDSM_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

###########################################################################################################

trainset="DDSM_CLAHE-mass_normal"
testset="MIAS_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done


########################################### DDSM+CBIS - Mass_ Normal #############################################################

trainset="DDSM+CBIS_CLAHE-mass_normal"
testset="DDSM+CBIS_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM+CBIS/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

###########################################################################################################

trainset="DDSM+CBIS_CLAHE-mass_normal"
testset="MIAS_CLAHE-mass_normal"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="mass_normal"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done


############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="DDSM_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="CMMD-only_mass"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_clahe \
    --clahe_clip_limit 5.0 \
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="CBIS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/CBIS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

############################################### Benign_Malignant - DDSM Train #######################################################
trainset="DDSM_CLAHE-benign_malignant"
testset="MIAS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

############################################### Benign_Malignant - DDSM-CBIS Train #######################################################
trainset="DDSM+CBIS_CLAHE-benign_malignant"
testset="DDSM+CBIS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/DDSM+CBIS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

###########################################################################################################
trainset="DDSM+CBIS_CLAHE-benign_malignant"
testset="CMMD-only_mass"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_clahe \
    --clahe_clip_limit 5.0 \
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

###########################################################################################################
trainset="DDSM+CBIS_CLAHE-benign_malignant"
testset="MIAS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/MIAS/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

############################################### Benign_Malignant - DDSM+CBIS+MIAS Train #######################################################
trainset="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
testset="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/zDiogo_Araujo/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done

###########################################################################################################
trainset="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
testset="CMMD-only_mass"
suffix_to_remove="/.ipynb_checkpoints"

datapath="../../data/CMMD/$testset"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    ckpt_path="$(find "Finetuned_Models/Binary/$classification_problem/$trainset/" -type d -path "*/Baselines-Finetune-$trainset-$base-*")"
    if [[ $ckpt_path == *$suffix_to_remove ]]; then
        ckpt_path="${ckpt_path%$suffix_to_remove}"
    fi
    ckpt_path="${ckpt_path}/Baseline-$base-best_checkpoint.pth"

    logdir="Baselines-Test-testset_$testset-trainset_$trainset-Date_$now"
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
    --breast_clahe \
    --clahe_clip_limit 5.0 \
    --breast_padding \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $trainset \
    --testset $testset \
    --data_path $datapath \
    --output_dir "Tests/Binary/$classification_problem/$testset/$logdir"
    
    echo "Output dir for the last experiment: $logdir"
done


exit 0