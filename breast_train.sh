datapath="../Data/ISIC2019bea_mel_nevus_limpo"
now=$(date +"%Y%m%d")

baselines=('resnet18' 'resnet50' 'vgg16'  'densenet169' 'efficientnet_b3' 'vit_b_16' 'deit_small_patch16_224' 'deit_base_patch16_224')
batch=64
n_classes=2
epoch=90

lr=2e-5
min_lr=2e-6
warmup_lr=4e-5
patience=12
delta=-0.1
sched='cosine'
drop=0.0
opt='adamw'


for base in "${baselines[@]}"
do

    logdir="BREAST-Baseline-Binary-$base-Date_$now"
    echo "----------------- Starting Program: $logdir --------------------"

    python3 main.py \
    --model "$base" \
    --nb_classes $n_classes \
    --project_name "MIA-Breast" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 8 \
    --epochs $epoch \
    --batch_size $batch \
    --input_size 224 \
    --lr $lr \
    --lr_scheduler \
    --lr_cycle_decay 0.8 \
    --min_lr $min_lr \
    --warmup_epochs 5 \
    --warmup_lr $warmup_lr \
    --patience $patience \
    --counter_saver_threshold 100 \
    --delta $delta \
    --weight-decay 1e-6 \
    --batch_aug \
    --loss_scaler \
    --data_path "$datapath" \
    --class_weights "balanced" \
    --dataset "DDSM+CBIS+MIAS_CLAHE-Binary" \
    --dataset_type "Breast" \
    --output_dir "Finetuned_Baselines/Binary/$logdir"

    echo "Output dir for the last experiment: $logdir"
done