################################ DDSM-Mass_vs_Normal #########################################

datapath="../Data/DDSM-Mass_vs_Normal"
dataset_name="DDSM-Mass_vs_Normal"
dataset_type="Breast"

baselines=('resnet18' 'resnet50' 'vgg16'  'densenet169' 'efficientnet_b3' 'vit_b_16' 'deit_small_patch16_224' 'deit_base_patch16_224')

batch=128
n_classes=2
epoch=90
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=30
delta=0.0
sched='cosine'
opt='adamw'
drops=(0.2)
drops_layers_rate=(0.0)
drop_block_rate=None
max_norm_grad=10.0
weight_decay=1e-6


for base in "${baselines[@]}"
do
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do

            now=$(date +"%Y%m%d_%H%M%S")
            logdir="Baselines-Finetune-$dataset_type-$dataset_name-$base-drop_$dropout-drop_layer_$drop_path-Date_$now"
            echo "----------------- Starting Program: $logdir --------------------"

            python main.py \
            --finetune \
            --model $base \
            --nb_classes $n_classes \
            --project_name "MIA-Breast" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:1" \
            --num_workers 8 \
            --epochs $epoch \
            --batch_size $batch \
            --input_size 224 \
            --lr_scheduler \
            --lr $lr \
            --min_lr $min_lr \
            --warmup_lr $warmup_lr \
            --lr_cycle_decay 0.8 \
            --classifier_warmup_epochs 5 \
            --warmup_epochs 10 \
            --patience $patience \
            --delta $delta \
            --counter_saver_threshold 100 \
            --weight-decay $weight_decay \
            --drop $dropout\
            --drop_layers_rate $drop_path \
            --loss_scaler \
            --clip_grad $max_norm_grad \
            --data_path $datapath \
            --class_weights "balanced" \
            --test_val_flag \
            --dataset $dataset_name \
            --dataset_type $dataset_type \
            --output_dir "Finetuned_Models/Binary/$dataset_name/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done

################################ DDSM-Benign_vs_Malignant #########################################

datapath="../Data/DDSM-Benign_vs_Malignant"
dataset_name="DDSM-Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18' 'resnet50' 'vgg16'  'densenet169' 'efficientnet_b3' 'vit_b_16' 'deit_small_patch16_224' 'deit_base_patch16_224')

batch=128
n_classes=2
epoch=90
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=30
delta=0.0
sched='cosine'
opt='adamw'
drops=(0.2)
drops_layers_rate=(0.0)
drop_block_rate=None
max_norm_grad=10.0
weight_decay=1e-6


for base in "${baselines[@]}"
do
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do

            now=$(date +"%Y%m%d_%H%M%S")
            logdir="Baselines-Finetune-$dataset_type-$dataset_name-$base-drop_$dropout-drop_layer_$drop_path-Date_$now"
            echo "----------------- Starting Program: $logdir --------------------"

            python main.py \
            --finetune \
            --model $base \
            --nb_classes $n_classes \
            --project_name "MIA-Breast" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:0" \
            --num_workers 8 \
            --epochs $epoch \
            --batch_size $batch \
            --input_size 224 \
            --lr_scheduler \
            --lr $lr \
            --min_lr $min_lr \
            --warmup_lr $warmup_lr \
            --lr_cycle_decay 0.8 \
            --classifier_warmup_epochs 5 \
            --warmup_epochs 10 \
            --patience $patience \
            --delta $delta \
            --counter_saver_threshold 100 \
            --weight-decay $weight_decay \
            --drop $dropout\
            --drop_layers_rate $drop_path \
            --loss_scaler \
            --clip_grad $max_norm_grad \
            --data_path $datapath \
            --class_weights "balanced" \
            --test_val_flag \
            --dataset $dataset_name \
            --dataset_type $dataset_type \
            --output_dir "Finetuned_Models/Binary/$dataset_name/$logdir"
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done