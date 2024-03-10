################################ DDSM-Benign_vs_Malignant  #########################################

datapath="../Data/DDSM-Benign_vs_Malignant"
dataset_name="DDSM-Benign_vs_Malignant"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18' 'resnet50' 'vgg16'  'densenet169' 'efficientnet_b3' 'vit_b_16' 'deit_small_patch16_224' 'deit_base_patch16_224' 'vit_small_patch16_224.augreg_in1k' 'vit_b_16')

batch=128
n_classes=2
epoch=140
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=45
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.2)
drop_block_rate=None
weight_decay=1e-6

for base in "${baselines[@]}"
do
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do
            for w_decay in "${weight_decay[@]}"
            do

                now=$(date +"%Y%m%d_%H%M%S")
                logdir="Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-drop_$dropout-drop_layer_$drop_path-weight_decay_$w_decay-Date_$now"
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
                --classifier_warmup_epochs 5 \
                --batch_size $batch \
                --input_size 224 \
                --sched $sched \
                --lr $lr \
                --min_lr $min_lr \
                --warmup_lr $warmup_lr \
                --warmup_epochs 10 \
                --patience $patience \
                --delta $delta \
                --counter_saver_threshold $epoch \
                --drop $dropout\
                --drop_layers_rate $drop_path \
                --weight-decay $weight_decay \
                --class_weights "balanced" \
                --test_val_flag \
                --loss_scaler \
                --dataset_type $dataset_type \
                --dataset $dataset_name \
                --data_path $datapath \
                --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
                
                echo "Output dir for the last experiment: $logdir"
            done
        done
    done
done

################################ DDSM-Mass_vs_Normal  #########################################

datapath="../Data/DDSM-Mass_vs_Normal"
dataset_name="DDSM-Mass_vs_Normal"
classification_problem="Mass_vs_Normal"
dataset_type="Breast"

baselines=('resnet18' 'resnet50' 'vgg16'  'densenet169' 'efficientnet_b3' 'vit_b_16' 'deit_small_patch16_224' 'deit_base_patch16_224' 'vit_small_patch16_224.augreg_in1k' 'vit_b_16')

batch=128
n_classes=2
epoch=140
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=45
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.2)
drop_block_rate=None
weight_decay=1e-6

for base in "${baselines[@]}"
do
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do
            for w_decay in "${weight_decay[@]}"
            do

                now=$(date +"%Y%m%d_%H%M%S")
                logdir="Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-drop_$dropout-drop_layer_$drop_path-weight_decay_$w_decay-Date_$now"
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
                --classifier_warmup_epochs 5 \
                --batch_size $batch \
                --input_size 224 \
                --sched $sched \
                --lr $lr \
                --min_lr $min_lr \
                --warmup_lr $warmup_lr \
                --warmup_epochs 10 \
                --patience $patience \
                --delta $delta \
                --counter_saver_threshold $epoch \
                --drop $dropout\
                --drop_layers_rate $drop_path \
                --weight-decay $weight_decay \
                --class_weights "balanced" \
                --test_val_flag \
                --loss_scaler \
                --dataset_type $dataset_type \
                --dataset $dataset_name \
                --data_path $datapath \
                --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
                
                echo "Output dir for the last experiment: $logdir"
            done
        done
    done
done