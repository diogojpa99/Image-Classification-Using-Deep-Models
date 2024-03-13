################################ DDSM-Benign_vs_Malignant  #########################################

datapath="../../../Data/ISIC2019bea_mel_nevus_limpo"
dataset_name="ISIC2019bea_mel_nevus_limpo"
classification_problem="Binary"
dataset_type="Breast"

baselines=('resnet18')

batch=128
n_classes=2
epoch=90
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=60
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.2)
drop_block_rate=None
weight_decay=1e-3

for base in "${baselines[@]}"
do
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do
            for w_decay in "${weight_decay[@]}"
            do

                now=$(date +"%Y%m%d_%H%M%S")
                logdir="Baselines-Finetune-$dataset_type-$classification_problem-$dataset_name-$base-Date_$now"
                echo "----------------- Starting Program: $logdir --------------------"

                python main.py \
                --finetune \
                --model $base \
                --nb_classes $n_classes \
                --project_name "Thesis" \
                --run_name "$logdir" \
                --hardware "Server" \
                --gpu "cuda:0" \
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
                --loss_scaler \
                --batch_aug \
                --dataset_type $dataset_type \
                --dataset $dataset_name \
                --data_path $datapath \
                --output_dir "wFinetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
                
                echo "Output dir for the last experiment: $logdir"
            done
        done
    done
done
