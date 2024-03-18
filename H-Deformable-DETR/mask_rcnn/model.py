from .backbone import build_backbone
from .network import MaskRCNN

def build_mask_rcnn(num_classes, args):
    backbone = build_backbone(args)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model