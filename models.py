import torch
import torch.nn as nn

import torchvision
from timm.models import create_model

cnns_baselines = ['resnet18', 'resnet50', 'vgg16', 'densenet169', 'efficientnet_b3']
transformers_baselines = ['deit_small_patch16_224', 'deit_base_patch16_224', 'vit_small_patch16_224.augreg_in1k', 'vit_b_16']
deits_baselines = ['deit_small_patch16_224', 'deit_base_patch16_224']

resnet_baselines = ['resnet18', 'resnet50']
other_cnn_baselines = ['vgg16', 'densenet169', 'efficientnet_b3']

def Define_Model(model:str, 
                 nb_classes:int, 
                 drop:float=0.0,
                 args=None) -> torch.nn.Module:
    """Defined the model to be used for training and testing. The model can be a CNN or a Transformer.

    Args:
        model (str): Name of the model to be used.
        nb_classes (int): Number of classes in the dataset.
        drop (float, optional): Dropout rate. Defaults to 0.0.
        args (_type_, optional): Arguments. Defaults to None.

    Raises:
        NotImplementedError: If the model is not implemented.

    Returns:
        torch.nn.Module: Model to be used for training and testing.
    """

    if model == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1) if args.finetune else torchvision.models.resnet18()
        model.fc = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(model.fc.in_features, nb_classes) 
        )
    elif model == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1) if args.finetune else torchvision.models.resnet50()
        model.fc = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(model.fc.in_features, nb_classes) 
        )
    elif model == 'vgg16': 
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1) if args.finetune else torchvision.models.vgg16()
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(model.classifier[-1].in_features, nb_classes) 
        )
    elif model == 'densenet169':
        model = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1) if args.finetune else torchvision.models.densenet169()
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(model.classifier.in_features, nb_classes) 
        )
    elif model == 'efficientnet_b3':
        model = create_model('efficientnet_b3', 
                             pretrained=True if args.finetune else False,
                             num_classes=nb_classes,
                             drop_rate=args.drop,  
                             drop_path_rate=args.drop_layers_rate)
        # model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nb_classes)
    elif model == 'vit_b_16':
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1) if args.finetune else torchvision.models.vit_b_16()
        model.heads.head = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(model.heads.head.in_features, nb_classes) 
        )
    elif model == 'vit_small_patch16_224.augreg_in1k':
        model = create_model(
            model,
            pretrained=True if args.finetune else False,
            num_classes=nb_classes,
            drop_rate=drop,
            pos_drop_rate=args.pos_drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_layers_rate,
            drop_block_rate=None,
            img_size=args.input_size,
            pos_encoding = args.pos_encoding_flag,
        )
    elif model == 'deit_small_patch16_224' or model == 'deit_base_patch16_224':
        model = create_model(
            model,
            pretrained=False,
            num_classes=nb_classes,
            drop_rate=drop,
            pos_drop_rate=args.pos_drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_layers_rate,
            img_size=args.input_size,
            pos_encoding = args.pos_encoding_flag,
        )

    else:
        raise NotImplementedError('This Baseline is not yet implemented!')
    
    return model
    
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
            