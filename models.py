from torch import nn
import torch
import torchvision.models as models

def conv2d_to_conv3d(layer):
    """
    Basically just takes the properties and weights of conv2d and
    converts to conv3d

    output:
    in_channels, out_channels: 64, 64
    kernel_size : (1,3,3) -> D x W x H
    stride: (1,1,1)
    padding: (0, 1, 1)
    """
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    new_weight = layer.weight.unsqueeze(2)
    kernel_size = tuple([1] + list(layer.kernel_size))
    stride = tuple([1] + list(layer.stride))
    padding = tuple([0] + list(layer.padding))

    new_layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    
    new_weight = layer.weight.unsqueeze(2)
    new_layer.weight.data = new_weight.data
    if layer.bias is not None:
        new_layer.bias.data = layer.bias.data
    
    return new_layer

def batch2d_to_batch3d(layer):
    """
    Converts batch normalization 2d layers into 3d
    """
    num_features = layer.num_features
    eps = layer.eps
    momentum = layer.momentum
    affine = layer.affine
    
    new_bn = nn.BatchNorm3d(num_features, eps, momentum, affine)
    new_bn.load_state_dict(layer.state_dict())
    
    return new_bn

def pool2d_to_pool3d(layer):

    kernel_size = tuple([1, layer.kernel_size, layer.kernel_size])
    stride = tuple([1, layer.stride, layer.stride])
    padding = layer.padding
    dilation = layer.dilation
    return_indices = layer.return_indices
    ceil_mode = layer.ceil_mode
    
    return nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

def transform_module_list(module_list):
    transformed_list = []
    
    for item in module_list:
        if isinstance(item, nn.Conv2d): 
            transformed_list.append(conv2d_to_conv3d(item))
        elif isinstance(item, nn.BatchNorm2d):
            transformed_list.append(batch2d_to_batch3d(item))
        elif isinstance(item, nn.MaxPool2d):
            transformed_list.append(pool2d_to_pool3d(item))
        elif isinstance(item, nn.ReLU):
            transformed_list.append(item)
        elif isinstance(item, nn.Sequential):
            transformed_list += transform_module_list(item.children())
        else:
            transformed_list.append(item)
            
    return transformed_list

def translate_block(block, downsample=False):
    block.conv1 = conv2d_to_conv3d(block.conv1)
    block.bn1 = batch2d_to_batch3d(block.bn1)
    
    block.conv2 = conv2d_to_conv3d(block.conv2)
    block.bn2 = batch2d_to_batch3d(block.bn2)
    
    if block.downsample is not None:
        conv = conv2d_to_conv3d(block.downsample[0])
        bn = batch2d_to_batch3d(block.downsample[1])
        block.downsample = nn.Sequential(*[conv, bn])
    
    return block

def translate_block_resnet50(block, downsample=False):
    block.conv1 = conv2d_to_conv3d(block.conv1)
    block.bn1 = batch2d_to_batch3d(block.bn1)
    
    block.conv2 = conv2d_to_conv3d(block.conv2)
    block.bn2 = batch2d_to_batch3d(block.bn2)

    block.conv3 = conv2d_to_conv3d(block.conv3)
    block.bn3 = batch2d_to_batch3d(block.bn3)
    
    if block.downsample is not None:
        conv = conv2d_to_conv3d(block.downsample[0])
        bn = batch2d_to_batch3d(block.downsample[1])
        block.downsample = nn.Sequential(*[conv, bn])
    
    return block

def conv1x3x3(in_, out):
    return nn.Conv3d(in_, out, kernel_size=(1,3,3), padding=(0,1,1))

def conv3x1x1(in_, out):
    """
    Not used unless depth argument specified in ConvRelu
    """
    # padding=(0,0,0) when depth reduction, otherwise (1,0,0)
    # return nn.Conv3d(in_, out, kernel_size=(3,1,1), padding=(0,0,0))
    return nn.Conv3d(in_, out, kernel_size=(3,1,1), padding=(1,0,0))

class ConvRelu(nn.Module):
    def __init__(self, in_, out, depth=False):
        super().__init__()
        if depth:
            self.conv = conv3x1x1(in_, out)
        else:
            self.conv = conv1x3x3(in_, out)
        self.bn = nn.BatchNorm3d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class ConvReluFromEnc(nn.Module):
    """
    Seems like its not in use

    """
    def __init__(self, conv, bn, relu):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Upsample2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels), #didnt specify depth, hence conv is (1,3,3), like resnet3d
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class AlbuNet3D34(nn.Module):
    """
    Credits to Jonas Wacker for the original implementation of AlbuNet3D
    """
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super(AlbuNet3D34, self).__init__()
        # resnet34 has a lot of blocks
        self.num_classes = num_classes
        self.encoder = models.resnet34(pretrained=pretrained)
     
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # kernel size decreased
        conv0_modules = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool]
        self.conv0 = nn.Sequential(*transform_module_list(conv0_modules))
        
        self.conv1_0 = translate_block(self.encoder.layer1[0])
        self.conv1_1 = translate_block(self.encoder.layer1[1])
        self.conv1_2 = translate_block(self.encoder.layer1[2])
        
        # number output filters: 64
        
        self.conv2_0 = translate_block(self.encoder.layer2[0])
        self.conv2_1 = translate_block(self.encoder.layer2[1])
        self.conv2_2 = translate_block(self.encoder.layer2[2])
        self.conv2_3 = translate_block(self.encoder.layer2[3])
        
        # number output filters: 128
        
        self.conv3_0 = translate_block(self.encoder.layer3[0])
        self.conv3_1 = translate_block(self.encoder.layer3[1])
        self.conv3_2 = translate_block(self.encoder.layer3[2])
        self.conv3_3 = translate_block(self.encoder.layer3[3])
        self.conv3_4 = translate_block(self.encoder.layer3[4])
        self.conv3_5 = translate_block(self.encoder.layer3[5])
        
        # number output filters: 256
        
        self.conv4_0 = translate_block(self.encoder.layer4[0])
        self.conv4_1 = translate_block(self.encoder.layer4[1])
        self.conv4_2 = translate_block(self.encoder.layer4[2])
        
        # number output filters: 512
        
        ### EVERYTHING ABOVE IS THE RESNET BACKBONE, USING RESNET WEIGHTS

        #############################################################################################
        # Albunet architecture starts below

        # self.center = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)
        self.center = ConvRelu(512, num_filters * 8) # 128 (512, 128)

        self.dec5 = Upsample2D(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (640, 256, 128)
        # self.dec5 = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (384, 256, 128)
        self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2) # (256, 128, 32)
        self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2) # (96, 64, 64)
        self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters) # (64, 64, 16)
        self.dec0 = ConvRelu(num_filters, num_filters) # (16,16)
        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1) # (16, 4, 1)
        
        self.depth1 = ConvRelu(512, 512, depth=True) # (512, 512)
        self.depth2 = ConvRelu(num_filters * 8, num_filters * 8, depth=True) # (128, 128)
        self.depth3 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True) # (64, 64)
        self.depth4 = ConvRelu(num_filters, num_filters, depth=True) # (16,16)
        
    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1_2(self.conv1_1(self.conv1_0(conv0)))
        conv2 = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(conv1))))
        conv3 = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(self.conv3_0(conv2))))))
        conv4 = self.conv4_2(self.conv4_1(self.conv4_0(conv3)))
        
        conv4 = self.depth1(conv4)
        
        center = self.center(conv4)

        dec5 = self.dec5(torch.cat([center, conv4], 1))
        dec4 = self.dec4(torch.cat([dec5, conv3], 1))
        dec4 = self.depth2(dec4)
        
        dec3 = self.dec3(torch.cat([dec4, conv2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))
        dec2 = self.depth3(dec2)
        
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        dec0 = self.depth4(dec0)

        if self.num_classes > 1:
            # x_out = F.log_softmax(self.final(dec0), dim=1)
            x_out = self.final(dec0) # softmax is moved to loss metric
        else:
            x_out = self.final(dec0)

        return x_out

class AlbuNet3D34_4channels(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False,updated=False):
        super(AlbuNet3D34_4channels, self).__init__()
        # resnet34 has a lot of blocks
        self.num_classes = num_classes
        self.encoder = models.resnet34(pretrained=pretrained)

        if updated:
            pretrained_weights =  self.encoder.conv1.weight.clone()
            self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.conv1.weight[:,:3].data = pretrained_weights
            self.encoder.conv1.weight[:, 3].data = self.encoder.conv1.weight[:, 0]

        else:
            print("Not upated, no weight initializations")
            self.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # kernel size decreased
        conv0_modules = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool]
        self.conv0 = nn.Sequential(*transform_module_list(conv0_modules))
        
        self.conv1_0 = translate_block(self.encoder.layer1[0])
        self.conv1_1 = translate_block(self.encoder.layer1[1])
        self.conv1_2 = translate_block(self.encoder.layer1[2])
        
        # number output filters: 64
        
        self.conv2_0 = translate_block(self.encoder.layer2[0])
        self.conv2_1 = translate_block(self.encoder.layer2[1])
        self.conv2_2 = translate_block(self.encoder.layer2[2])
        self.conv2_3 = translate_block(self.encoder.layer2[3])
        
        # number output filters: 128
        
        self.conv3_0 = translate_block(self.encoder.layer3[0])
        self.conv3_1 = translate_block(self.encoder.layer3[1])
        self.conv3_2 = translate_block(self.encoder.layer3[2])
        self.conv3_3 = translate_block(self.encoder.layer3[3])
        self.conv3_4 = translate_block(self.encoder.layer3[4])
        self.conv3_5 = translate_block(self.encoder.layer3[5])
        
        # number output filters: 256
        
        self.conv4_0 = translate_block(self.encoder.layer4[0])
        self.conv4_1 = translate_block(self.encoder.layer4[1])
        self.conv4_2 = translate_block(self.encoder.layer4[2])
        
        # number output filters: 512
        
        ### EVERYTHING ABOVE IS THE RESNET BACKBONE, USING RESNET WEIGHTS

        #############################################################################################
        # Albunet architecture starts below

        # self.center = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)
        self.center = ConvRelu(512, num_filters * 8) # 128 (512, 128)

        self.dec5 = Upsample2D(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (640, 256, 128)
        # self.dec5 = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (384, 256, 128)
        self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2) # (256, 128, 32)
        self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2) # (96, 64, 64)
        self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters) # (64, 64, 16)
        self.dec0 = ConvRelu(num_filters, num_filters) # (16,16)
        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1) # (16, 4, 1)
        
        self.depth1 = ConvRelu(512, 512, depth=True)
        self.depth2 = ConvRelu(num_filters * 8, num_filters * 8, depth=True) # (128, 128)
        self.depth3 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True) # (64, 64)
        self.depth4 = ConvRelu(num_filters, num_filters, depth=True) # (16,16)
        
    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1_2(self.conv1_1(self.conv1_0(conv0)))
        conv2 = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(conv1))))
        conv3 = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(self.conv3_0(conv2))))))
        conv4 = self.conv4_2(self.conv4_1(self.conv4_0(conv3)))
        
        conv4 = self.depth1(conv4)
        
        center = self.center(conv4)

        dec5 = self.dec5(torch.cat([center, conv4], 1))
        dec4 = self.dec4(torch.cat([dec5, conv3], 1))
        dec4 = self.depth2(dec4)
        
        dec3 = self.dec3(torch.cat([dec4, conv2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))
        dec2 = self.depth3(dec2)
        
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        dec0 = self.depth4(dec0)

        if self.num_classes > 1:
            # x_out = F.log_softmax(self.final(dec0), dim=1)
            x_out = self.final(dec0) # softmax is moved to loss metric
        else:
            x_out = self.final(dec0)

        return x_out

##%  Unused model architectures
class AlbuNet3D34_UPDATED(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super(AlbuNet3D34_UPDATED, self).__init__()
        # resnet34 has a lot of blocks
        self.num_classes = num_classes

        # self.encoder = models.resnet34(pretrained=pretrained)
        self.encoder = models.resnet50(pretrained=pretrained)
     
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # kernel size decreased
        conv0_modules = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool]
        self.conv0 = nn.Sequential(*transform_module_list(conv0_modules))
        
        self.conv1_0 = translate_block_resnet50(self.encoder.layer1[0])
        self.conv1_1 = translate_block_resnet50(self.encoder.layer1[1])
        self.conv1_2 = translate_block_resnet50(self.encoder.layer1[2])

        # number output filters: 256
        
        self.conv2_0 = translate_block_resnet50(self.encoder.layer2[0])
        self.conv2_1 = translate_block_resnet50(self.encoder.layer2[1])
        self.conv2_2 = translate_block_resnet50(self.encoder.layer2[2])
        self.conv2_3 = translate_block_resnet50(self.encoder.layer2[3])
        
        # number output filters: 512
        
        self.conv3_0 = translate_block_resnet50(self.encoder.layer3[0])
        self.conv3_1 = translate_block_resnet50(self.encoder.layer3[1])
        self.conv3_2 = translate_block_resnet50(self.encoder.layer3[2])
        self.conv3_3 = translate_block_resnet50(self.encoder.layer3[3])
        self.conv3_4 = translate_block_resnet50(self.encoder.layer3[4])
        self.conv3_5 = translate_block_resnet50(self.encoder.layer3[5])
        
        # number output filters: 1024
        
        self.conv4_0 = translate_block_resnet50(self.encoder.layer4[0])
        self.conv4_1 = translate_block_resnet50(self.encoder.layer4[1])
        self.conv4_2 = translate_block_resnet50(self.encoder.layer4[2])
        
        # number output filters: 2048
        
        ### EVERYTHING ABOVE IS THE RESNET BACKBONE OR PRETRAINED BACKBONE, USING RESNET WEIGHTS

        #############################################################################################
        # Albunet architecture starts below

        # self.center = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)

        self.center = ConvRelu(2048, num_filters * 8) # 128 (512, 128)

        self.dec5 = Upsample2D(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (640, 256, 128)
        self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (384, 256, 128)
        self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2) # (256, 128, 32)
        self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2) # (96, 64, 64)
        self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters) # (64, 64, 16)
        self.dec0 = ConvRelu(num_filters, num_filters) # (16,16)

        # self.dec7 = Upsample2D(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (640, 256, 128)
        # self.dec6 = Upsample2D(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (384, 256, 128)
        # self.dec5 = Upsample2D(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (640, 256, 128)
        # self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (384, 256, 128)
        # self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2) # (256, 128, 32)
        # self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2) # (96, 64, 64)
        # self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters) # (64, 64, 16)
        # self.dec0 = ConvRelu(num_filters, num_filters) # (16,16)

        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1) # (16, 4, 1)

        # self.depth1 = ConvRelu(2048, 2048, depth=True) # (2048, 2048)
        # self.depth2 = ConvRelu(num_filters * 32, num_filters * 32, depth=True) # (512,512)
        # self.depth3 = ConvRelu(num_filters * 4 * 4, num_filters * 4 * 4, depth=True) # (256, 256)
        # self.depth4 = ConvRelu(num_filters * 2 * 4, num_filters * 2 * 4, depth=True) # (128, 128)
        # self.depth5 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True) # (64, 64)
        # self.depth6 = ConvRelu(num_filters, num_filters, depth=True) # (16, 16)
        

        self.depth1 = ConvRelu(2048, 2048, depth=True)
        self.depth2 = ConvRelu(num_filters * 8, num_filters * 8, depth=True) # (128, 128)
        self.depth3 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True) # (64, 64)
        self.depth4 = ConvRelu(num_filters, num_filters, depth=True) # (16, 16)

        # self.depth1 = ConvRelu(512, 512, depth=True)
        # self.depth2 = ConvRelu(num_filters * 8, num_filters * 8, depth=True) # (128, 128)
        # self.depth3 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True) # (64, 64)
        # self.depth4 = ConvRelu(num_filters, num_filters, depth=True) # (16, 16)
        
    def forward(self, x):
        
        # These are to connect the resnet layers
        conv0 = self.conv0(x)
        conv1 = self.conv1_2(self.conv1_1(self.conv1_0(conv0)))
        conv2 = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(conv1))))
        conv3 = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(self.conv3_0(conv2))))))
        conv4 = self.conv4_2(self.conv4_1(self.conv4_0(conv3)))
        
        conv4 = self.depth1(conv4)
        center = self.center(conv4)
        # END HERE FOR ENCODER SECTION 

        # START HERE FOR DECODER LAYERS

        dec5 = self.dec5(torch.cat([center, conv4], 1))
        dec4 = self.dec4(torch.cat([dec5, conv3],1))
        dec4 = self.depth2(dec4)

        dec3 = self.dec3(torch.cat([dec4, conv2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))
        dec2 = self.depth3(dec2)

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        dec0 = self.depth4(dec0)

        if self.num_classes > 1:
            # x_out = F.log_softmax(self.final(dec0), dim=1)
            x_out = self.final(dec0) # softmax is moved to loss metric
        else:
            x_out = self.final(dec0)

        return x_out

class AlbuNet3D34_UPDATED_COLAB(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=False, is_deconv=False):
        super(AlbuNet3D34_UPDATED_COLAB, self).__init__()
        # resnet34 has a lot of blocks
        self.num_classes = num_classes

        # self.encoder = models.resnet34(pretrained=pretrained)
        self.encoder = models.resnet50(pretrained=pretrained)
     
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) # kernel size decreased
        conv0_modules = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool]
        self.conv0 = nn.Sequential(*transform_module_list(conv0_modules))
        
        self.conv1_0 = translate_block_resnet50(self.encoder.layer1[0])
        self.conv1_1 = translate_block_resnet50(self.encoder.layer1[1])
        self.conv1_2 = translate_block_resnet50(self.encoder.layer1[2])

        # number output filters: 256
        
        self.conv2_0 = translate_block_resnet50(self.encoder.layer2[0])
        self.conv2_1 = translate_block_resnet50(self.encoder.layer2[1])
        self.conv2_2 = translate_block_resnet50(self.encoder.layer2[2])
        self.conv2_3 = translate_block_resnet50(self.encoder.layer2[3])
        
        # number output filters: 512
        
        self.conv3_0 = translate_block_resnet50(self.encoder.layer3[0])
        self.conv3_1 = translate_block_resnet50(self.encoder.layer3[1])
        self.conv3_2 = translate_block_resnet50(self.encoder.layer3[2])
        self.conv3_3 = translate_block_resnet50(self.encoder.layer3[3])
        self.conv3_4 = translate_block_resnet50(self.encoder.layer3[4])
        self.conv3_5 = translate_block_resnet50(self.encoder.layer3[5])
        
        # number output filters: 1024
        
        self.conv4_0 = translate_block_resnet50(self.encoder.layer4[0])
        self.conv4_1 = translate_block_resnet50(self.encoder.layer4[1])
        self.conv4_2 = translate_block_resnet50(self.encoder.layer4[2])
        
        # number output filters: 2048
        
        ### EVERYTHING ABOVE IS THE RESNET BACKBONE OR PRETRAINED BACKBONE, USING RESNET WEIGHTS

        #############################################################################################
        # Albunet architecture starts below

        # self.center = Upsample2D(512, num_filters * 8 * 2, num_filters * 8)

        self.center = ConvRelu(2048, num_filters * 8) # 128 (512, 128)

        # self.dec5 = Upsample2D(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (640, 256, 128)
        # self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) # (384, 256, 128)
        # self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2) # (256, 128, 32)
        # self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2) # (96, 64, 64)
        # self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters) # (64, 64, 16)
        # self.dec0 = ConvRelu(num_filters, num_filters) # (16,16)

        self.dec7 = Upsample2D(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) 
        self.dec6 = Upsample2D(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) 
        self.dec5 = Upsample2D(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) 
        self.dec4 = Upsample2D(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8) 
        self.dec3 = Upsample2D(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2) 
        self.dec2 = Upsample2D(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2) 
        self.dec1 = Upsample2D(num_filters * 2 * 2, num_filters * 2 * 2, num_filters) 
        self.dec0 = ConvRelu(num_filters, num_filters) 

        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1) # (16, 4, 1)

        self.depth1 = ConvRelu(2048, 2048, depth=True) # (2048, 2048)
        self.depth2 = ConvRelu(num_filters * 8, num_filters * 8, depth=True) # (256,256)
        self.depth3 = ConvRelu(num_filters * 4 * 2, num_filters * 4 * 2, depth=True) # (128, 128)
        self.depth4 = ConvRelu(num_filters * 2 * 2, num_filters * 2 * 2, depth=True) # (64, 64)
        self.depth5 = ConvRelu(num_filters, num_filters, depth=True) # (16, 16)

    def forward(self, x):
        
        # These are to connect the resnet layers
        conv0 = self.conv0(x)
        conv1 = self.conv1_2(self.conv1_1(self.conv1_0(conv0)))
        conv2 = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(conv1))))
        conv3 = self.conv3_5(self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(self.conv3_0(conv2))))))
        conv4 = self.conv4_2(self.conv4_1(self.conv4_0(conv3)))
        
        conv4 = self.depth1(conv4)

        center = self.center(conv4)
        # END HERE FOR ENCODER SECTION 

        # START HERE FOR DECODER LAYERS
        dec7 = self.dec7(torch.cat([center, conv4], 1))

        dec6 = self.dec6(torch.cat([dec7, conv3],1))
        dec6 = self.depth2(dec6) #error here

        dec5 = self.dec5(torch.cat([dec6, conv2], 1))
        dec4 = self.dec4(torch.cat([dec5, conv1],1))
        dec4 = self.depth3(dec4)

        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec2 = self.depth4(dec2)

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        dec0 = self.depth5(dec0)

        if self.num_classes > 1:
            # x_out = F.log_softmax(self.final(dec0), dim=1)
            x_out = self.final(dec0) # softmax is moved to loss metric
        else:
            x_out = self.final(dec0)

        return x_out