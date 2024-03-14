import torch
import torch.nn as nn
import torch.nn.functional as F

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
        model.fc = nn.Linear(model.fc.in_features, nb_classes) 
    elif model == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1) if args.finetune else torchvision.models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, nb_classes) 
    elif model == 'vgg16': 
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1, dropout=drop) if args.finetune else torchvision.models.vgg16(dropout=drop)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nb_classes)
    elif model == 'densenet169':
        model = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1, drop_rate=drop) if args.finetune else torchvision.models.densenet169(drop_rate=drop)
        model.classifier = nn.Linear(model.classifier.in_features, nb_classes)
    elif model == 'efficientnet_b3':
        model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1, dropout=drop) if args.finetune else torchvision.models.efficientnet_b3(dropout=drop)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nb_classes)
    elif model == 'vit_b_16':
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1, dropout=drop) if args.finetune else torchvision.models.vit_b_16(dropout=drop)
        model.heads.head = nn.Linear(model.heads.head.in_features, nb_classes)
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

class SimplifiedCNN(nn.Module):
    def __init__(self, nb_classes=2, input_size=224, drop=0.5):
        super(SimplifiedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 512) # Adjusted based on the output size after pooling
        self.dropout1 = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(512, nb_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 56 * 56) # Adjust the flattening based on the model architecture
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, 
                 nb_classes:int=2, 
                 input_size:int=224,
                 drop:float=0.0):
        
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.dropout1 = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(1024, nb_classes) 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 256 * 7 * 7) 
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)  
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ComplexCNN(nn.Module):
    def __init__(self, nb_classes=2):
        super(ComplexCNN, self).__init__()
        
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Configurations for scalable architecture
        blocks_per_layer = [2, 2, 2, 2]
        channels_per_layer = [64, 128, 256, 512, 1024]
        
        self.layers = nn.ModuleList()
        self.in_channels = 64
        
        # Creating scalable layers
        for idx, num_blocks in enumerate(blocks_per_layer):
            out_channels = channels_per_layer[idx]
            stride = 1 if idx == 0 else 2
            self.layers.append(self._make_layer(out_channels, num_blocks, stride))

        # Reduction layers before classifier
        self.reduction = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, nb_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.initial_bn(self.initial_conv(x))))

        for layer in self.layers:
            x = layer(x)

        x = self.reduction(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
