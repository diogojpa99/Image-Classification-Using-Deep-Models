import models, utils, engine

import breast_scripts.data_setup as data_setup


import torch
import torch.backends.cudnn as cudnn

from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma, NativeScaler
from timm.scheduler import create_scheduler
import torch.optim as optim

import argparse
from pathlib import Path
import datetime
import time
import numpy as np
import wandb
import warnings
from sklearn.exceptions import UndefinedMetricWarning   

from typing import List, Union

from sklearn.model_selection import KFold
import numpy as np
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler

import os

def cross_validation(data_path:str,
                     n_splits:int=5,
                     seed:int=42,
                     device:str="cuda:0",
                     wandb=print,
                     args=None) -> list:
    """_summary_

    Args:
        device (_type_, optional): _description_. Defaults to "cuda:0".
        wandb (_type_, optional): _description_. Defaults to print.
        args (_type_, optional): _description_. Defaults to None.

    Returns:
        list: _description_
    """
    
    # For k_fold cross validation, we will use the training set
    data_path = os.path.join(data_path, 'train')
    
    # Define K-Fold Cross Validation    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    
    # Define datasets
    dataset = ImageFolder(root=data_path, transform=None, loader=None)
    
    train_transform = data_setup.Train_Transform(args.input_size, args)
    val_transform = data_setup.Test_Transform(args.input_size, args)    

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        
        print(f"[Info] Starting fold {fold+1}")
        
        ################## Data Setup ##################
        
        # Build the datasets
        if args.breast_loader=='Gray_PIL_Loader_Wo_He':
            train_set = ImageFolder(root=data_path, transform=train_transform, loader=data_setup.Gray_PIL_Loader_Wo_He)
            val_set = ImageFolder(root=data_path, transform=val_transform, loader=data_setup.Gray_PIL_Loader_Wo_He)
        elif args.breast_loader=='Gray_PIL_Loader_Wo_He_No_Resize':
            train_set = ImageFolder(root=data_path, transform=train_transform, loader=data_setup.Gray_PIL_Loader_Wo_He_No_Resize)
            val_set = ImageFolder(root=data_path, transform=val_transform, loader=data_setup.Gray_PIL_Loader_Wo_He_No_Resize)
        else:
            train_set = ImageFolder(root=data_path, transform=train_transform, loader=data_setup.Gray_PIL_Loader)
            val_set = ImageFolder(root=data_path, transform=val_transform, loader=data_setup.Gray_PIL_Loader)
               
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        print(f"[Info] Train set size: {len(train_idx)} | Val set size: {len(val_idx)}")
        
        data_loader_train = torch.utils.data.DataLoader(
            train_set, sampler=train_subsampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            val_set, sampler=val_subsampler,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        
        ############################ Define the Feature Extractor ############################
        
        if args.model is None:
            model = models.SimplifiedCNN(nb_classes=args.nb_classes, drop=args.drop)
            #model = models.SimpleCNN(nb_classes=args.nb_classes, drop=args.drop)
            #model = models.ComplexCNN(nb_classes=args.nb_classes)
        else:
            model = models.Define_Model(model=args.model, nb_classes=args.nb_classes, drop=args.drop, args=args)
            if args.finetune and (args.model in models.deits_baselines) and not args.from_pretrained_baseline_flag:
                args.pretrained_baseline_path = models.Pretrained_Baseline_Paths(args.model, args)
                if args.pretrained_baseline_path:
                    utils.Load_Pretrained_Baseline(args.pretrained_baseline_path, model, args)
            elif args.finetune and args.from_pretrained_baseline_flag and (args.pretrained_baseline_path is not None):
                print(f"[Info] Loading the pretrained model from:\n'{args.pretrained_baseline_path}'")
                utils.Load_Finetuned_Baseline(path=args.pretrained_baseline_path, model=model, args=args)
        
        if args.resume:
            print(f"[Info] Loading the finetuned model from:\n'{args.resume}'")
            utils.Load_Finetuned_Baseline(path=args.resume, model=model, args=args)
                
        model.to(device)
        
        ############################ Define the Model EMA ############################
        model_ema = None 
        if args.model_ema:
            model_ema = ModelEma(model,decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
            #model_ema.ema.to(device)
                
        ################## Define Training Parameters ##################
        
        # Define the output directory
        output_dir = Path(args.output_dir + f'/fold_{fold+1}')
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
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
            if args.sched == 'exp':
                lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
            else:    
                lr_scheduler,_ = create_scheduler(args, optimizer)
            
            # (4) Define the loss function with class weighting
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
        ########################## Training or evaluating ###########################
        
        if args.resume:
            
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
                engine.Classifier_Warmup(model, epoch, args.classifier_warmup_epochs, args)
                
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
            
                if lr_scheduler is not None:
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
                
                if epoch % 10 == 0:
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
                        if args.sched is not None:
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
            
            #utils.plot_loss_and_acc_curves(train_results, val_results, output_dir=output_dir, args=args)
            
        #utils.plot_confusion_matrix(best_results["confusion_matrix"], train_set.class_to_idx, output_dir=output_dir, args=args)
                
        print('\n---------------- Val. stats for the best model ----------------\n',
            f"Acc: {best_results['acc1']} | Bacc: {best_results['bacc']} | F1-score: {np.mean(best_results['f1_score'])} | \n",
            f"Class-to-idx: {train_set.class_to_idx} | \n",
            f"Precisions: {best_results['precision']} | \n",
            f"Recalls: {best_results['recall']} | \n")
        
        fold_results.append(best_results)
        
        if wandb!=print:
            wandb.log({"Best Val. Acc": best_results['acc1'], "Best Val. Bacc": best_results['bacc'], "Best Val. F1-score": np.mean(best_results['f1_score'])})
            wandb.log({"Training time": total_time_str})
            #wandb.finish()
            
    for idx, result in enumerate(fold_results):
        print(f"Fold {idx+1} - Best Val. Acc: {result['acc1']:.4f} | Best Val. Bacc: {result['bacc']:.4f} | Best Val. F1-score: {np.mean(result['f1_score']):.4f}")
            
    if wandb!=print:
        wandb.finish()
        
    return
            
        
        