"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from timm.utils import ModelEma

from typing import Optional
from collections import Counter

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, \
    balanced_accuracy_score
    
import models
      
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               loss_scaler,
               max_norm: float=0.0,
               lr_scheduler=None,
               wandb=print,
               model_ema: Optional[ModelEma] = None,
               args = None):
        
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    train_stats = {}
    lr_num_updates = epoch * len(dataloader)
    preds = []; targs = []

    # Loop through data loader data batches
    for batch_idx, (input, target) in enumerate(dataloader):
        
        # Send data to device
        input, target = input.to(device), target.to(device)
        
        # 1. Clear gradients
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(input) 
            loss = criterion(output, target)
        train_loss += loss.item() 
        
        if loss_scaler is not None:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order # this attribute is added by timm on one optimizer (adahessian)
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward() 
            optimizer.step() 

        # Update LR Scheduler
        if not args.cosine_one_cycle:
            lr_scheduler.step_update(num_updates=lr_num_updates)
            
        # Update Model Ema
        if model_ema is not None:
            if device == 'cuda:0' or device == 'cuda:1':
                torch.cuda.synchronize()
            model_ema.update(model)

        # Calculate and accumulate accuracy metric across all batches
        predictions = torch.argmax(output, dim=1)
        train_acc += (predictions == target).sum().item()/len(predictions)
        
        preds.append(predictions.cpu().numpy()); targs.append(target.cpu().numpy())
        
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader); train_acc = train_acc / len(dataloader)

    train_stats['train_loss'] = train_loss
    train_stats['train_acc'] = train_acc
    train_stats['train_lr'] = optimizer.param_groups[0]['lr']
    
    if wandb!=print:
        wandb.log({"Train Loss":train_loss}, step=epoch)
        wandb.log({"Train Accuracy":train_acc},step=epoch)
        wandb.log({"Train LR":optimizer.param_groups[0]['lr']},step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds); targs=np.concatenate(targs)
    train_stats['confusion_matrix'], train_stats['f1_score'] = confusion_matrix(targs, preds), f1_score(targs, preds, average=None) 
    train_stats['precision'], train_stats['recall'] = precision_score(targs, preds, average=None), recall_score(targs, preds, average=None)
    train_stats['bacc'] = balanced_accuracy_score(targs, preds)
    train_stats['acc1'], train_stats['loss'] = train_acc, train_loss
    
    return train_stats

@torch.no_grad()
def evaluation(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               criterion: torch.nn.Module, 
               device: torch.device,
               epoch: int,
               wandb=print,
               args=None):
    
    # Switch to evaluation mode
    model.eval()
    
    preds = []
    targs = []
    test_loss, test_acc = 0, 0
    results = {}
    deit_cls_dist = {'mean':[], 'mel':[], 'nv':[]}
    
    for input, target in dataloader:
        
        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
        # Compute output
        output = model(input) 
        loss = criterion(output, target)
        test_loss += loss.item()
    
        # Calculate and accumulate accuracy
        predictions = torch.argmax(output, dim=1)
        test_acc += ((predictions == target).sum().item()/len(predictions))
        
        preds.append(predictions.cpu().numpy()); targs.append(target.cpu().numpy())
        
        if args.model in models.deits_baselines and args.visualize_cls_token:
            models.cls_token_dist(model, deit_cls_dist, predictions,args)
            
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss/len(dataloader); test_acc = test_acc/len(dataloader)

    if wandb!=print:
        wandb.log({"Val Loss":test_loss},step=epoch)
        wandb.log({"Val Accuracy":test_acc},step=epoch)
        
    # Compute Metrics
    preds=np.concatenate(preds); targs=np.concatenate(targs)
    results['confusion_matrix'], results['f1_score'] = confusion_matrix(targs, preds), f1_score(targs, preds, average=None) 
    results['precision'], results['recall'] = precision_score(targs, preds, average=None), recall_score(targs, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targs, preds)
    results['acc1'], results['loss'] = accuracy_score(targs, preds), test_loss

    return results, deit_cls_dist

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score - self.delta:
            # If we don't have an improvement, increase the counter 
            self.counter += 1
            #self.trace_func(f'\tEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # If we have an imporvement, save the model
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            #self.trace_func(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model as checkpoint.pth')
            self.trace_func(f'\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def Class_Weighting(train_set:torch.utils.data.Dataset, 
                    val_set:torch.utils.data.Dataset, 
                    device:str='cuda:0', 
                    args=None):
    """ Class weighting for imbalanced datasets.

    Args:
        train_set (torch.utils.data.Dataset): Training set.
        val_set (torch.utils.data.Dataset): Validation set.
        device (str): Device to use.
        args (*args): Arguments.

    Returns:
        torch.Tensor: Class weights. (shape: (2,))
    """
    train_dist = dict(Counter(train_set.targets))
    val_dist = dict(Counter(val_set.targets))
            
    if args.class_weights == 'median':
        class_weights = torch.Tensor([(len(train_set)/x) for x in train_dist.values()]).to(device)
    else:                   
        class_weights = torch.Tensor(compute_class_weight(class_weight=args.class_weights, 
                                                        classes=np.unique(train_set.targets), y=train_set.targets)).to(device)

    print(f"Classes map: {train_set.class_to_idx}"); print(f"Train distribution: {train_dist}"); print(f"Val distribution: {val_dist}")
    print(f"Class weights: {class_weights}\n")
    
    return class_weights

def Classifier_Warmup(model: torch.nn.Module, 
                      current_epoch: int, 
                      warmup_epochs: int, 
                      args=None):    
    """Function that defines if we are in the warmup phase or not.

    Args:
        model (torch.nn.Module): _description_
        current_epoch (int): _description_
        warmup_epochs (int): _description_
        flag (bool): _description_
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    if current_epoch==0 and warmup_epochs>0:
        for param in model.parameters():
            param.requires_grad = False
            
        if args.model in models.resnet_baselines:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif args.model in models.other_cnn_baselines:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.model == 'vit_b_16':
            for param in model.heads.head.parameters():
                param.requires_grad = True
        elif args.model in models.transformers_baselines:
            for param in model.head.parameters():
                param.requires_grad = True
                
        trainable_params={name: param for name, param in model.named_parameters() if param.requires_grad}
        print(f"[Info] - Warmup phase: Only the head is trainable.")
        #print(f"Trainable parameters: {trainable_params.keys()}.")
    elif current_epoch == warmup_epochs:
        for param in model.parameters():
            param.requires_grad = True    
        trainable_params={name: param for name, param in model.named_parameters() if param.requires_grad}
        print(f"[Info] - Finetune phase: All parameters are trainable.")
        #print(f"Trainable parameters: {trainable_params.keys()}.")
        