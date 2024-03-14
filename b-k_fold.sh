################################ DDSM-Benign_vs_Malignant  #########################################

datapath="../../data/zDiogo_Araujo/CBIS-DDSM-train_val-pad_clahe"
dataset_name="CBIS-DDSM-train_val-pad_clahe"
classification_problem="Benign_vs_Malignant"
dataset_type="Breast"

baselines=('resnet18')

batch=256
n_classes=2
epoch=190
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=120
delta=0.0
sched='cosine'
optimizers=('adamw')
drops=(0.3)
drops_layers_rate=(0.0)
drop_block_rate=None
weight_decay=1e-3
loader="Gray_PIL_Loader_Wo_He_No_Resize"

for base in "${baselines[@]}"
do
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do
            for w_decay in "${weight_decay[@]}"
            do

                now=$(date +"%Y%m%d_%H%M%S")
                logdir="Baselines-K_Fold-$dataset_type-$classification_problem-$dataset_name-$base-loader_$loader-Date_$now"
                echo "----------------- Starting Program: $logdir --------------------"

                python main.py \
                --finetune \
                --model $base \
                --nb_classes $n_classes \
                --project_name "MIA-Breast" \
                --run_name "$logdir" \
                --hardware "Server" \
                --gpu "cuda:0" \
                --num_workers 12 \
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
                --breast_loader $loader \
                --kfold \
                --kfold_split 10 \
                --dataset_type $dataset_type \
                --dataset $dataset_name \
                --data_path $datapath \
                --output_dir "wFinetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
                
                echo "Output dir for the last experiment: $logdir"
            done
        done
    done
done

exit 0