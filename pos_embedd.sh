############# EviT Experiments to see what is the best keep rate COM Fuse token ##############

datapath="../Data/Bea_LIMPO/limpo"
keep_rates=(1.0)
drop_loc="(3, 6, 9)"
lr=2e-4

for kr in "${keep_rates[@]}"
do 
    now=$(date +"%Y%m%d")
    ckpt="EViT/Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    logdir="EViT_Small-Multiclass_SoftAug-kr_$kr-FuseToken-DropLoc_Default-lr_init_$lr-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"

    python3 EViT/main.py \
    --model deit_small_patch16_shrink_base \
    --nb_classes 8 \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --finetune $ckpt \
    --batch-size 256 \
    --epochs 90 \
    --input-size 224 \
    --base_keep_rate $kr \
    --fuse_token \
    --drop_loc "$drop_loc" \
    --drop 0.1 \
    --lr_scheduler \
    --lr $lr \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --weight-decay 1e-6 \
    --shrink_start_epoch 0 \
    --warmup_epochs 0 \
    --shrink_epochs 0 \
    --patience 100 \
    --counter_saver_threshold 100 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler \
    --data-path "$datapath" \
    --output_dir "EViT/Finetuned_Models/Multiclass/kr_$kr/$logdir"

done


################################## Binary ##################################

datapath="../Data/ISIC2019bea_mel_nevus_limpo"
now=$(date +"%Y%m%d")
n_class=2
lr=2e-4

###### Baselines ######

baselines=('vit_s_16' 'deit_small_patch16_224' )

for base in "${baselines[@]}"
do
    logdir="Baseline-$base-Pos_Embed_OFF-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"
    
    if [ $base == 'vit_s_16' ]
    then
        bas='vit_small_patch16_224.augreg_in1k'
    else
        bas=$base
    fi

    python3 Baselines/main.py \
    --model $bas \
    --pos_encoding_flag \
    --nb_classes $n_class \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --num_workers 12 \
    --batch_size 128 \
    --epochs 90 \
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
    --output_dir "Pos_Embedd_Off/Binary/Baselines/$logdir"

    echo "Output dir for the last experiment: $logdir"
done


########## EViT ##########

keep_rates=(0.5 0.7 0.8 0.9 1.0)
drop_loc="(3, 6, 9)"

for kr in "${keep_rates[@]}"
do 
    now=$(date +"%Y%m%d")
    
    if [ $kr==0.5 ] 
    then
        ckpt="EViT/Pretrained_Models/evit-0.5-fuse-img224-deit-s.pth"
    elif [ $kr==0.6 ] 
    then
        ckpt="EViT/Pretrained_Models/evit-0.6-fuse-img224-deit-s.pth"
    elif [ $kr==0.7]
    then
        ckpt="EViT/Pretrained_Models/evit-0.7-fuse-img224-deit-s.pth"
    elif [ $kr==0.8]
    then
        ckpt="EViT/Pretrained_Models/evit-0.8-fuse-img224-deit-s.pth"
    else
        ckpt="EViT/Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    fi

    logdir="EViT_Small-kr_$kr-Pos_Embed_OFF-DropLoc_Default-lr_init_$lr-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"

    python3 EViT/main.py \
    --model deit_small_patch16_shrink_base \
    --nb_classes $n_class \
    --pos_encoding_flag \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --finetune $ckpt \
    --batch-size 256 \
    --epochs 50 \
    --input-size 224 \
    --base_keep_rate $kr \
    --drop_loc "$drop_loc" \
    --drop 0.1 \
    --lr_scheduler \
    --lr $lr \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --weight-decay 1e-6 \
    --shrink_start_epoch 0 \
    --warmup_epochs 0 \
    --shrink_epochs 0 \
    --patience 100 \
    --counter_saver_threshold 100 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler \
    --data-path "$datapath" \
    --output_dir "Pos_Embedd_Off/Binary/EViT/kr_$kr/$logdir"

done


########## MIL ############

patch_extractors=('deitSmall' 'evitSmall_07')
mil_types=('instance' 'embedding')
pooling_types=('max' 'avg' 'topk')

#### MIL - EViT_small_07 #####

for patch_ext in "${patch_extractors[@]}"
do

    if [ $patch_ext == 'deitSmall' ]
    then
        k=49
        feat_ext='deit_small_patch16_224'
        ckpt='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
    else
        k=17
        feat_ext='deit_small_patch16_shrink_base'
        ckpt='MIL/Feature_Extractors/Pretrained_EViTs/evit-0.7-img224-deit-s.pth'
    fi

    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do

            logdir="MIL-$patch_ext-$mil_t-$pool-lr_init_$lr-Time_$now"
            echo "----------------- Output dir: $logdir --------------------"

            python3 MIL/main.py \
            --project_name "Thesis" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:1" \
            --feature_extractor $feat_ext \
            --pretrained_feature_extractor_path $ckpt \
            --nb_classes $n_class \
            --pos_encoding_flag \
            --drop 0.1 \
            --num_workers 12 \
            --batch_size 128 \
            --epochs 90 \
            --input_size 224 \
            --mil_type $mil_t \
            --pooling_type $pool \
            --lr $lr \
            --lr_scheduler \
            --lr_cycle_decay 0.8 \
            --min_lr 2e-4 \
            --warmup_epochs 5 \
            --warmup_lr 1e-4 \
            --weight-decay 1e-6 \
            --patience 200 \
            --counter_saver_threshold 100 \
            --batch_aug \
            --loss_scaler \
            --data_path "$datapath" \
            --topk $k \
            --output_dir "Pos_Embedd_Off/Binary/MIL/$patch_ext/$logdir"

            echo "Output dir for the last experiment: $logdir"
            
        done
    done
done



################################## Multiclass ##################################

datapath="../Data/Bea_LIMPO/limpo"
now=$(date +"%Y%m%d")
n_class=8

###### Baselines ######

baselines=('vit_s_16' 'deit_small_patch16_224' )


for base in "${baselines[@]}"
do
    logdir="Baseline-Multiclass_SoftAug-$base-Pos_Embed_OFF-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"
    
    if [ $base == 'vit_s_16' ]
    then
        bas='vit_small_patch16_224.augreg_in1k'
    else
        bas=$base
    fi

    python3 Baselines/main.py \
    --model $bas \
    --pos_encoding_flag \
    --nb_classes $n_class  \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --num_workers 12 \
    --batch_size 128 \
    --epochs 90 \
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
    --output_dir "Pos_Embedd_Off/Multiclass/Baselines/$logdir"

    echo "Output dir for the last experiment: $logdir"
done

########## EViT ##########

keep_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
drop_loc="(3, 6, 9)"

for kr in "${keep_rates[@]}"
do 
    now=$(date +"%Y%m%d")
    
    if [ $kr==0.5 ] 
    then
        ckpt="EViT/Pretrained_Models/evit-0.5-fuse-img224-deit-s.pth"
    elif [ $kr==0.6 ] 
    then
        ckpt="EViT/Pretrained_Models/evit-0.6-fuse-img224-deit-s.pth"
    elif [ $kr==0.7]
    then
        ckpt="EViT/Pretrained_Models/evit-0.7-fuse-img224-deit-s.pth"
    elif [ $kr==0.8]
    then
        ckpt="EViT/Pretrained_Models/evit-0.8-fuse-img224-deit-s.pth"
    else
        ckpt="EViT/Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    fi


    logdir="EViT_Small-kr_$kr-Multiclass_SoftAug-Pos_Embed_OFF-DropLoc_Default-lr_init_$lr-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"

    python3 EViT/main.py \
    --model deit_small_patch16_shrink_base \
    --nb_classes $n_class \
    --pos_encoding_flag \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --finetune $ckpt \
    --batch-size 256 \
    --epochs 50 \
    --input-size 224 \
    --base_keep_rate $kr \
    --drop_loc "$drop_loc" \
    --drop 0.1 \
    --lr_scheduler \
    --lr $lr \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --weight-decay 1e-6 \
    --shrink_start_epoch 0 \
    --warmup_epochs 0 \
    --shrink_epochs 0 \
    --patience 100 \
    --counter_saver_threshold 100 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler \
    --data-path "$datapath" \
    --output_dir "Pos_Embedd_Off/Multiclass/EViT/kr_$kr/$logdir"

done



#### MIL  #####

patch_extractors=('deitSmall' 'evitSmall_07')
mil_types=('instance')
pooling_types=('max' 'avg' 'topk')

#### Method 1

for patch_ext in "${patch_extractors[@]}"
do

    if [ $patch_ext == 'deitSmall' ]
    then
        k=49
        feat_ext='deit_small_patch16_224'
        ckpt='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
    else
        k=17
        feat_ext='deit_small_patch16_shrink_base'
        ckpt='MIL/Feature_Extractors/Pretrained_EViTs/evit-0.7-img224-deit-s.pth'
    fi

    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do

            logdir="MIL-Multiclass_SoftAug-Method_first-$patch_ext-$mil_t-$pool-lr_init_$lr-Time_$now"
            echo "----------------- Output dir: $logdir --------------------"

            python3 MIL/main.py \
            --project_name "Thesis" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:1" \
            --feature_extractor $feat_ext \
            --pretrained_feature_extractor_path $ckpt \
            --nb_classes $n_class \
            --pos_encoding_flag \
            --drop 0.1 \
            --num_workers 12 \
            --batch_size 128 \
            --epochs 100 \
            --input_size 224 \
            --mil_type $mil_t \
            --pooling_type $pool \
            --lr $lr \
            --lr_scheduler \
            --lr_cycle_decay 0.8 \
            --min_lr 2e-4 \
            --warmup_epochs 5 \
            --warmup_lr 1e-4 \
            --weight-decay 1e-6 \
            --patience 200 \
            --counter_saver_threshold 100 \
            --batch_aug \
            --loss_scaler \
            --data_path "$datapath" \
            --topk $k \
            --multiclass_method "first" \
            --output_dir "Pos_Embedd_Off/Multiclass/MIL/Method_1/$patch_ext/$logdir"

            echo "Output dir for the last experiment: $logdir"
            
        done
    done
done


#### Method 2

for patch_ext in "${patch_extractors[@]}"
do

    if [ $patch_ext == 'deitSmall' ]
    then
        k=49
        feat_ext='deit_small_patch16_224'
        ckpt='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
    else
        k=17
        feat_ext='evit_small_patch16_shrink_base'
        ckpt='MIL/Feature_Extractors/Pretrained_EViTs/evit-0.7-img224-deit-s.pth'
    fi

    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do

            logdir="MIL-Multiclass_SoftAug-Method_second-$patch_ext-$mil_t-$pool-lr_init_$lr-Time_$now"
            echo "----------------- Output dir: $logdir --------------------"

            python3 MIL/main.py \
            --project_name "Thesis" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:1" \
            --feature_extractor $feat_ext \
            --pretrained_feature_extractor_path $ckpt \
            --nb_classes $n_class \
            --pos_encoding_flag \
            --drop 0.1 \
            --num_workers 12 \
            --batch_size 128 \
            --epochs 100 \
            --input_size 224 \
            --mil_type $mil_t \
            --pooling_type $pool \
            --lr $lr \
            --lr_scheduler \
            --lr_cycle_decay 0.8 \
            --min_lr 2e-4 \
            --warmup_epochs 5 \
            --warmup_lr 1e-4 \
            --weight-decay 1e-6 \
            --patience 200 \
            --counter_saver_threshold 100 \
            --batch_aug \
            --loss_scaler \
            --data_path "$datapath" \
            --topk $k \
            --multiclass_method "second" \
            --output_dir "Pos_Embedd_Off/Multiclass/MIL/Method_2/$patch_ext/$logdir"

            echo "Output dir for the last experiment: $logdir"
            
        done
    done
done

#### Embedding

mil_types=('embedding')

for patch_ext in "${patch_extractors[@]}"
do

    if [ $patch_ext == 'deitSmall' ]
    then
        k=49
        feat_ext='deit_small_patch16_224'
        ckpt='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
    else
        k=17
        feat_ext='evit_small_patch16_shrink_base'
        ckpt='MIL/Feature_Extractors/Pretrained_EViTs/evit-0.7-img224-deit-s.pth'
    fi

    for mil_t in "${mil_types[@]}"
    do
        for pool in "${pooling_types[@]}"
        do

            logdir="MIL-Multiclass_SoftAug-$patch_ext-$mil_t-$pool-lr_init_$lr-Time_$now"
            echo "----------------- Output dir: $logdir --------------------"

            python3 MIL/main.py \
            --project_name "Thesis" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:1" \
            --feature_extractor $feat_ext \
            --pretrained_feature_extractor_path $ckpt \
            --nb_classes $n_class \
            --pos_encoding_flag \
            --drop 0.1 \
            --num_workers 12 \
            --batch_size 128 \
            --epochs 100 \
            --input_size 224 \
            --mil_type $mil_t \
            --pooling_type $pool \
            --lr $lr \
            --lr_scheduler \
            --lr_cycle_decay 0.8 \
            --min_lr 2e-4 \
            --warmup_epochs 5 \
            --warmup_lr 1e-4 \
            --weight-decay 1e-6 \
            --patience 200 \
            --counter_saver_threshold 100 \
            --batch_aug \
            --loss_scaler \
            --data_path "$datapath" \
            --topk $k \
            --output_dir "Pos_Embedd_Off/Multiclass/MIL/$patch_ext/$logdir"

            echo "Output dir for the last experiment: $logdir"
            
        done
    done
done