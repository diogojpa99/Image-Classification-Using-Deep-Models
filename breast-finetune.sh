####### General Finetuning Settings #######
baselines=('efficientnet_b3' 'deit_small_patch16_224')

batch=128
n_classes=2
epoch=90
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=20
delta=0.0
sched='cosine'
optimizers=('adamw')
dropout=0.3
drop_path=0.2
weight_decay=1e-4
loader="Gray_PIL_Loader_Wo_He_No_Resize"


############################################################## DDSM - Mass vs. Normal ####################################################################

datapath="../../data/zDiogo_Araujo/DDSM/DDSM_CLAHE-mass_normal"
dataset_name="DDSM_CLAHE-mass_normal"
classification_problem="mass_normal"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    logdir="Baselines-Finetune-$dataset_name-$base-$now"
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
    --drop $dropout \
    --drop_layers_rate $drop_path \
    --weight-decay $weight_decay \
    --class_weights "balanced" \
    --test_val_flag \
    --loss_scaler \
    --breast_loader $loader \
    --breast_padding \
    --breast_strong_aug \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"

done

############################################################## DDSM+CBIS - Mass vs. Normal ####################################################################


datapath="../../data/zDiogo_Araujo/DDSM+CBIS/DDSM+CBIS_CLAHE-mass_normal"
dataset_name="DDSM+CBIS_CLAHE-mass_normal"
classification_problem="mass_normal"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    logdir="Baselines-Finetune-$dataset_name-$base-$now"
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
    --drop $dropout \
    --drop_layers_rate $drop_path \
    --weight-decay $weight_decay \
    --class_weights "balanced" \
    --test_val_flag \
    --loss_scaler \
    --breast_loader $loader \
    --breast_padding \
    --breast_strong_aug \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"

done

############################################################## DDSM - Benign vs. Malignant ####################################################################

datapath="../../data/zDiogo_Araujo/DDSM/DDSM_CLAHE-benign_malignant"
dataset_name="DDSM_CLAHE-benign_malignant"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    logdir="Baselines-Finetune-$dataset_name-$base-$now"
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
    --drop $dropout \
    --drop_layers_rate $drop_path \
    --weight-decay $weight_decay \
    --class_weights "balanced" \
    --test_val_flag \
    --loss_scaler \
    --breast_loader $loader \
    --breast_padding \
    --breast_strong_aug \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"

done

############################################################## DDSM+CBIS - Benign vs. Malignant ####################################################################
datapath="../../data/zDiogo_Araujo/DDSM+CBIS/DDSM+CBIS_CLAHE-benign_malignant"
dataset_name="DDSM+CBIS_CLAHE-benign_malignant"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    logdir="Baselines-Finetune-$dataset_name-$base-$now"
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
    --drop $dropout \
    --drop_layers_rate $drop_path \
    --weight-decay $weight_decay \
    --class_weights "balanced" \
    --test_val_flag \
    --loss_scaler \
    --breast_loader $loader \
    --breast_padding \
    --breast_strong_aug \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"

done

############################################################## DDSM+CBIS+MIAS_CLAHE - Benign vs. Malignant ####################################################################
datapath="../../data/zDiogo_Araujo/DDSM+CBIS+MIAS_CLAHE-benign_malignant"
dataset_name="DDSM+CBIS+MIAS_CLAHE-benign_malignant"
classification_problem="benign_malignant"
dataset_type="Breast"

for base in "${baselines[@]}"
do

    now=$(date +"%Y%m%d_%H%M%S")
    logdir="Baselines-Finetune-$dataset_name-$base-$now"
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
    --drop $dropout \
    --drop_layers_rate $drop_path \
    --weight-decay $weight_decay \
    --class_weights "balanced" \
    --test_val_flag \
    --loss_scaler \
    --breast_loader $loader \
    --breast_padding \
    --breast_strong_aug \
    --breast_transform_rgb \
    --dataset_type $dataset_type \
    --dataset $dataset_name \
    --data_path $datapath \
    --output_dir "Finetuned_Models/Binary/$classification_problem/$dataset_name/$logdir"
    
    echo "Output dir for the last experiment: $logdir"

done

exit 0