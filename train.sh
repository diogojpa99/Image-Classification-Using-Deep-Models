datapath="../Data/ISIC2019bea_mel_nevus_limpo"
now=$(date +"%Y%m%d")

baselines=('resnet18' 'resnet50' 'vgg16'  'densenet169' 'efficientnet_b3' 'vit_b_16' 'deit_small_patch16_224' 'deit_base_patch16_224')
lr=2e-4
batch=256

for base in "${baselines[@]}"
do
    logdir="Baseline-$bas-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"
    if [ "$base" == "densenet169" ] || [ "$base" == "efficientnet_b3" ] || [ "$base" == "vit_b_16" ] || [ "$base" == "deit_base_patch16_224" ]; then
        batch=128
    else
        batch=256
    fi
    
    python3 main.py \
    --model "$base" \
    --nb_classes 2 \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --num_workers 12 \
    --batch_size $batch \
    --epochs 110 \
    --input_size 224 \
    --lr $lr \
    --lr_scheduler \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --warmup_lr 1e-6 \
    --weight-decay 1e-6 \
    --patience 100 \
    --counter_saver_threshold 100 \
    --batch_aug \
    --loss_scaler \
    --data_path "$datapath" \
    --output_dir "Finetuned-Baselines/Binary/$logdir"

    echo "Output dir for the last experiment: $logdir"
done