import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

cnns_baselines = ['resnet18.tv_in1k', 'resnet50.tv_in1k', 'vgg16.tv_in1k', 'densenet169.tv_in1k', 'efficientnet_b3']
transformers_baselines = ['deit_small_patch16_224', 'deit_base_patch16_224', 'vit_small_patch16_224.augreg_in1k', 'vit_b_16']
deits_baselines = ['deit_small_patch16_224', 'deit_base_patch16_224']

def Pretrained_Baseline_Paths(model, args) -> str: 
    """Selects the right checkpoint for the selected feature extractor model.
    
    Args:
        model (str): Name of the feature extractor model.
        args (**): Arguments from the parser.
    Returns:
        str: Path to the checkpoint of the feature extractor model.
    """
    checkpoints = ['https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
                   'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
                   'https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
                   ]
    
    if model == "deit_small_patch16_224":
        return checkpoints[0]
    elif model == "deit_base_patch16_224":
        return checkpoints[1]
    elif model == "vit_small_patch16_224.augreg_in1k":
        return checkpoints[2]
    else:
        raise ValueError(f"Invalid model: {model}. Must be 'resnet18.tv_in1k',\
            'resnet50.tv_in1k', 'deit_small_patch16_224', 'deit_base_patch16_224', 'vgg16.tv_in1k', 'efficientnet' or 'deit_small_patch16_shrink_base'")
        
def cls_token_dist(model, cls_attn, predictions, args):
    """During inference visualize the softax values of the attention of the CLS token.

    Args:
        model (torch.nn.module): model.
        args (kargs*): arguments from the parser.
        
    returns:
        None
    """
    
    # (1) Get the attention weights of the CLS token from the last Transformer block
    attn_map = model.blocks[-1].attn.get_attn_map()
    attn_map = torch.mean(attn_map, dim=1) # Average over the heads
    attn_of_cls = attn_map[:, 0]

    # Saving the average attention of the CLS token
    cls_attn['mean'].append(torch.mean(attn_of_cls, dim=0))
    
    # Saving the average attention of the CLS token for predicted MEL class (MEL == 0)
    pred_mel_idx = torch.where(predictions == 0)[0]
    mel_attn_cls = torch.gather(input=attn_of_cls, dim=0, index=pred_mel_idx.unsqueeze(-1).expand(-1,attn_of_cls.shape[1]))
    cls_attn['mel'].append(torch.mean(mel_attn_cls,dim=0))
    
    # Saving the average attention of the CLS token for predicted NV class (NV == 1)
    pred_nv_idx = torch.where(predictions == 1)[0]
    nv_attn_cls = torch.gather(input=attn_of_cls, dim=0, index=pred_nv_idx.unsqueeze(-1).expand(-1,attn_of_cls.shape[1]))
    cls_attn['nv'].append(torch.mean(nv_attn_cls,dim=0))
            