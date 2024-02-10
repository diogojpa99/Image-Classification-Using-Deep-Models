import Models.DEiT as deit, Models.ViT as vit

import models, utils, data_setup, engine

import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torchvision

from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma, NativeScaler
from timm.scheduler import create_scheduler
from timm.models import create_model
import torch.optim as optim

import argparse
from pathlib import Path
import datetime
import time
import numpy as np
import wandb

from typing import List, Union

import os
#os.environ["WANDB_MODE"] = "offline"

def get_args_parser():
   
    parser = argparse.ArgumentParser('Baselines', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='Output', help='path where to save, empty for no saving')
    parser.add_argument('--data_path', default='', help='path to input file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default='cuda:1', help='GPU id to use.')

    parser.add_argument('--train', action='store_true', default=False, help='Training mode.')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode.')
    parser.add_argument('--finetune', action='store_true', default=False, help='finetune mode.')
    parser.add_argument('--infer', action='store_true', default=False, help='Inference mode.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')

    # Dataset
    parser.add_argument('--dataset', default='ISIC2019-Clean', type=str, 
                        choices=['ISIC2019-Clean', 'PH2', 'Derm7pt','DDSM+CBIS+MIAS_CLAHE-Binary-Mass_vs_Normal', 
                                 'DDSM+CBIS+MIAS_CLAHE-Binary-Benign_vs_Malignant', 'DDSM+CBIS+MIAS_CLAHE', 
                                 'DDSM+CBIS+MIAS_CLAHE-v2', 'INbreast'], metavar='DATASET')
    parser.add_argument('--dataset_type', default='Skin', type=str, choices=['Breast', 'Skin'], metavar='DATASET')
    
    # Wanb parameters
    parser.add_argument('--project_name', default='Thesis', help='name of the project')
    parser.add_argument('--hardware', default='Server', choices=['Server', 'Colab', 'MyPC'], help='hardware used')
    parser.add_argument('--run_name', default='Baselines', help='name of the run')
    parser.add_argument('--wandb_flag', action='store_false', default=True, help='whether to use wandb')
    
    # Data parameters
    parser.add_argument('--input_size', default=224, type=int, help='image size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--nb_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='pin memory')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--classifier_warmup_epochs', type=int, default=5, metavar='N')
                
    # Baselines parameters
    parser.add_argument('--model', default='resnet18', type=str, metavar='MODEL',
                        choices=['resnet18', 'resnet50','vgg16', 'densenet169', 'efficientnet_b3', 'vit_small_patch16_224.augreg_in1k', 
                                 'vit_b_16', 'deit_small_patch16_224', 'deit_base_patch16_224',], 
                        help='Feature Extractor model architecture (default: "resnet18")')
    
    parser.add_argument('--pretrained_baseline_path', default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth', type=str, 
                        choices=['https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
                                 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
                                 'https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',], 
                        metavar='PATH', help="Download the pretrained feature extractor from the given path.")
    
    parser.add_argument('--baseline_pretrained_dataset', default='ImageNet1k', type=str, metavar='DATASET')

    # Evaluation parameters
    parser.add_argument('--evaluate_model_name', default='Baseline.pth', type=str, help="")
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
        
    # Imbalanced dataset parameters
    parser.add_argument('--class_weights', default='None', choices=['None', 'balanced', 'median'], type=str, 
                        help="Class weights for loss function.")
    
    # Optimizer parameters 
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adamw', 'sgd'],
                        help='Optimizer (default: "adamw")')
    
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters 
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', choices=['step', 'multistep', 'cosine', 'plateau','poly', 'exp'],
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    
    # * Lr Cosine Scheduler Parameters
    parser.add_argument('--cosine_one_cycle', type=bool, default=False, help='Only use cosine one cycle lr scheduler')
    parser.add_argument('--lr_k_decay', type=float, default=1.0, help='LR k rate (default: 1.0)')
    parser.add_argument('--lr_cycle_mul', type=float, default=1.0, help='LR cycle mul (default: 1.0)')
    parser.add_argument('--lr_cycle_decay', type=float, default=1.0, help='LR cycle decay (default: 1.0)')
    parser.add_argument('--lr_cycle_limit', type=int, default=1, help= 'LR cycle limit(default: 1)')
    
    parser.add_argument('--lr-noise', type=Union[float, List[float]], default=None, help='Add noise to lr')
    parser.add_argument('--lr-noise-pct', type=float, default=0.1, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.1)')
    parser.add_argument('--lr-noise-std', type=float, default=0.05, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 0.05)')
    
    # * Warmup parameters
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, metavar='LR',
                        help='warmup learning rate (default: 1e-3)')
    
    parser.add_argument('--min_lr', type=float, default=1e-4, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='Epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='Patience epochs for Plateau LR scheduler (default: 10.')

    # * StepLR parameters
    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR.')
    
    # * MultiStepLRScheduler parameters
    parser.add_argument('--decay_milestones', type=List[int], nargs='+', default=(10, 15), 
                        help='Epochs at which to decay learning rate.')
    
    # * The decay rate is transversal to many schedulers | However it has a different meaning for each scheduler
    # MultiStepLR: decay factor of learning rate | PolynomialLR: power factor | ExpLR: decay factor of learning rate
    parser.add_argument('--decay_rate', '--dr', type=float, default=1., metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Model EMA parameters -> Exponential Moving Average Model
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=12, metavar='N')
    parser.add_argument('--delta', type=float, default=0.0, metavar='N')
    parser.add_argument('--counter_saver_threshold', type=int, default=12, metavar='N')
    
    # Data augmentation parameters 
    parser.add_argument('--batch_aug', action='store_true', default=False, help='whether to augment batch')
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT', help='Color jitter factor (default: 0.)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + \
                        "(default: rand-m9-mstd0.5-inc1)'),
    
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.1, metavar='PCT', help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const', help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')
    
    # Loss scaler parameters
    parser.add_argument('--loss_scaler', action='store_true', default=False, help='Use loss scaler')
    
    # Deit Cls Token Visualization
    parser.add_argument('--visualize_cls_token', action='store_true', default=False, help='Visualize the attention weights of the CLS token.')
    parser.add_argument('--pos_encoding_flag', action='store_false', default=True, help='Whether to use positional encoding or not.')
    
    # Breast Data setup parameters
    parser.add_argument('--loader', default='Gray_PIL_Loader_Wo_Her', type=str, metavar='LOADER', choices=['Gray_PIL_Loader', 'Gray_PIL_Loader_Wo_He'])
    parser.add_argument('--test_val_flag', default=True, type=bool, help='If True, the test set is used as the validation set.')
    
    # Dropout parameters
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the classification head (default: 0.)')
    parser.add_argument('--pos_drop_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate for the positional encoding (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate for the attention layers (default: 0.)')
    parser.add_argument('--drop_layers_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate for the layers (default: 0.)')
    parser.add_argument('--drop_block_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate for the blocks (default: 0.)')
        
    return parser


def main(args):
    
    if not args.train and not args.eval and not args.finetune and not args.infer:
        raise ValueError('The mode is not specified. Please specify the mode: --train, --eval, --finetune, --infer.')
    
    # Start a new wandb run to track this script
    if args.wandb_flag:
        wandb.init(
            project=args.project_name,
            #mode="offline",
            config={
            "Baseline model": args.model,
            "Baseline dataset": args.baseline_pretrained_dataset,
            "Dataset": args.dataset,
            "epochs": args.epochs,"batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs, "Warmup lr": args.warmup_lr,
            "cooldown_epochs": args.cooldown_epochs, "patience_epochs": args.patience_epochs,
            "lr_scheduler": args.sched, "lr": args.lr, "min_lr": args.min_lr,
            "drop": args.drop, "weight_decay": args.weight_decay,
            "optimizer": args.opt, "momentum": args.momentum,
            "seed": args.seed, "class_weights": args.class_weights,
            "early_stopping_patience": args.patience, "early_stopping_delta": args.delta,
            "model_ema": args.model_ema, "Batch_augmentation": args.batch_aug, "Loss_scaler": args.loss_scaler,
            "PC": args.hardware,
            }
        )
        wandb.run.name = args.run_name
        
    # if args.debug:
    #     wandb=print
    
    if args.train or args.finetune: # Print arguments
        print("----------------- Args -------------------")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print("------------------------------------------\n")
            
    device = args.gpu if torch.cuda.is_available() else "cpu" # Set device
    print(f"Device: {device}\n")
    
    utils.configure_seed(args.seed) # Fix the seed for reproducibility
    cudnn.benchmark = True
    
    ################## Data Setup ##################
    if args.data_path:
        
        if not args.infer:
            train_set, val_set = data_setup.Build_Dataset(data_path = args.data_path, input_size=args.input_size, args=args)
            
            ## Data Loaders 
            sampler_train = torch.utils.data.RandomSampler(train_set)
            sampler_val = torch.utils.data.SequentialSampler(val_set)
            
            data_loader_train = torch.utils.data.DataLoader(
                train_set, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
            data_loader_val = torch.utils.data.DataLoader(
                val_set, sampler=sampler_val,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
     
    ############################ Define the Feature Extractor ############################
    
    model = models.Define_Model(model=args.model, nb_classes=args.nb_classes, drop=args.drop, args=args)
    model.to(device)
    
    if args.finetune and args.model in models.deits_baselines:
        args.pretrained_baseline_path = models.Pretrained_Baseline_Paths(args.model, args)
        if args.pretrained_baseline_path:
            utils.Load_Pretrained_Baseline(args.pretrained_baseline_path, model, args)
            
    ############################ Define the Model EMA ############################
    model_ema = None 
    if args.model_ema:
        model_ema = ModelEma(model,decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
            
    ################## Define Training Parameters ##################
    
    # Define the output directory
    output_dir = Path(args.output_dir)
        
    if args.data_path:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        print(f"Number of parameters: {n_parameters}\n")
        
        # (1) Define the class weights
        class_weights = engine.Class_Weighting(train_set, val_set, device, args)
        
        # (2) Define the optimizer
        optimizer = create_optimizer(args=args, model=model)

        # Define the loss scaler
        loss_scaler = NativeScaler() if args.loss_scaler else None

        # (3) Create scheduler
        if args.lr_scheduler:
            if args.sched == 'exp':
                lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
            else:    
                lr_scheduler,_ = create_scheduler(args, optimizer)
        
        # (4) Define the loss function with class weighting
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
    ########################## Training or evaluating ###########################
    
    if args.resume:
        utils.Load_Finetuned_Baseline(path=args.resume, model=model, args=args)
        
        if args.eval:
            print('******* Starting evaluation process. *******')
            total_time_str = 0
            best_results, deit_cls_vis = engine.evaluation(model=model,
                                                           dataloader=data_loader_val,
                                                           criterion=torch.nn.CrossEntropyLoss(), 
                                                           epoch=0, 
                                                           device=device,
                                                           args=args)
            
            if args.visualize_cls_token and args.model in models.deits_baselines:
                utils.Visualize_cls_token_dist(model, deit_cls_vis, args)
                
        elif args.infer:
            print('Still to be implemented.')
            # TODO: Add inference code
            # Receive an input image
            # Infer with the already finetuned model
            # Return the prediction
            # Note: Should define its own inference_loader, and so on
                                
    elif args.train or args.finetune:
        
        start_time = time.time()  
        train_results = {'loss': [], 'acc': [] , 'lr': []}
        val_results = {'loss': [], 'acc': [], 'f1': [], 'cf_matrix': [], 'bacc': [], 'precision': [], 'recall': []}
        best_val_bacc = 0.0; best_results = None
        early_stopping = engine.EarlyStopping(patience=args.patience, verbose=True, delta=args.delta, path=str(output_dir) +'/checkpoint.pth')
        
        if not args.pos_encoding_flag and args.model in models.transformers_baselines:
            for i, (param_name, param) in enumerate(model.named_parameters()):
                if param_name == 'pos_embed':
                    param.requires_grad = False
                    break 
        
        print(f"******* Start training for {(args.epochs + args.cooldown_epochs)} epochs. *******") 
        for epoch in range(args.start_epoch, (args.epochs + args.cooldown_epochs)):
            
            # Classifier Warmup
            engine.Classifier_Warmup(model, epoch, args.warmup_epochs, args)
            
            train_stats = engine.train_step(model=model,
                                            dataloader=data_loader_train,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            device=device,  
                                            epoch=epoch+1,
                                            wandb=wandb,
                                            loss_scaler=loss_scaler,
                                            max_norm=args.clip_grad,
                                            lr_scheduler=lr_scheduler,
                                            model_ema=model_ema,
                                            args=args)
        
            if args.lr_scheduler:
                lr_scheduler.step(epoch+1)

            results,_ = engine.evaluation(model=model,
                                        dataloader=data_loader_val,
                                        criterion=criterion,
                                        device=device,
                                        epoch=epoch+1,
                                        wandb=wandb,
                                        args=args) 
            
            # Update results dictionary
            train_results['loss'].append(train_stats['train_loss']); train_results['acc'].append(train_stats['train_acc']); train_results['lr'].append(train_stats['train_lr'])
            val_results['acc'].append(results['acc1']); val_results['loss'].append(results['loss']); val_results['f1'].append(results['f1_score'])
            val_results['cf_matrix'].append(results['confusion_matrix']); val_results['precision'].append(results['precision'])
            val_results['recall'].append(results['recall']); val_results['bacc'].append(results['bacc'])
            
            print(f"Epoch: {epoch+1} | lr: {train_stats['train_lr']:.5f} | Train Loss: {train_stats['train_loss']:.4f} | Train Acc: {train_stats['train_acc']:.4f} |",
                  f"Val. Loss: {results['loss']:.4f} | Val. Acc: {results['acc1']:.4f} | Val. Bacc: {results['bacc']:.4f} | F1-score: {np.mean(results['f1_score']):.4f}")
                            
            if results['bacc'] > best_val_bacc and early_stopping.counter < args.counter_saver_threshold:
                # Only want to save the best checkpoints if the best val bacc and the early stopping counter is less than the threshold
                best_val_bacc = results['bacc']
                checkpoint_paths = [output_dir / f'Baseline-{args.model}-best_checkpoint.pth']
                best_results = results
                for checkpoint_path in checkpoint_paths:
                    checkpoint_dict = {
                        'model':model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.lr_scheduler:
                        checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
                    if model_ema is not None:
                        checkpoint_dict['model_ema'] = get_state_dict(model_ema)
                    utils.save_on_master(checkpoint_dict, checkpoint_path)
                print(f"\tBest Val. Bacc: {(best_val_bacc*100):.2f}% |[INFO] Saving model as 'best_checkpoint.pth'")
                        
            # Early stopping
            early_stopping(results['loss'], model)
            if early_stopping.early_stop:
                print("\t[INFO] Early stopping - Stop training")
                break
         
        # Compute the total training time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        print('\n---------------- Train stats for the last epoch ----------------\n',
            f"Acc: {train_stats['acc1']:.3f} | Bacc: {train_stats['bacc']:.3f} | F1-score: {np.mean(train_stats['f1_score']):.3f} | \n",
            f"Class-to-idx: {train_set.class_to_idx} | \n",
            f"Precisions: {best_results['precision']} | \n",
            f"Recalls: {best_results['recall']} | \n",
            f"Confusion Matrix: {train_stats['confusion_matrix']}\n",
            f"Training time {total_time_str}\n")
        
        utils.plot_loss_and_acc_curves(train_results, val_results, output_dir=output_dir, args=args)

    utils.plot_confusion_matrix(best_results["confusion_matrix"], train_set.class_to_idx, output_dir=output_dir, args=args)
            
    print('\n---------------- Val. stats for the best model ----------------\n',
        f"Acc: {best_results['acc1']} | Bacc: {best_results['bacc']} | F1-score: {np.mean(best_results['f1_score'])} | \n",
        f"Class-to-idx: {train_set.class_to_idx} | \n",
        f"Precisions: {best_results['precision']} | \n",
        f"Recalls: {best_results['recall']} | \n")
    
    if wandb!=print:
        wandb.log({"Best Val. Acc": best_results['acc1'], "Best Val. Bacc": best_results['bacc'], "Best Val. F1-score": np.mean(best_results['f1_score'])})
        wandb.log({"Training time": total_time_str})
        #wandb.finish()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Baselines', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)