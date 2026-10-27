from models.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        # the ImageNet weights are not downloaded: TagLab always loads the full DeepLab weights afterwards
        # (deeplab-resnet.pth.tar for training, the .net file for classification)
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=False)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
