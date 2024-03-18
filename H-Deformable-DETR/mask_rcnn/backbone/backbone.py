import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from ..train_utils.misc import is_main_process
from .swin_transformer import SwinTransformer
from .feature_pyramid_network import IntermediateLayerGetter, FeaturePyramidNetwork, LastLevelMaxPool


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list):
        out = self.body(tensor_list)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer,
        )
        assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class SwinBackbone(nn.Module):
    def __init__(
        self, backbone: str, args
    ):
        super().__init__()
        if backbone == "swin_tiny":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.2,
                patch_norm=True,
                use_checkpoint=True,
            )
            backbone.init_weights(args.teacher_pretrain_backbone)
        elif backbone == "swin_small":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.2,
                patch_norm=True,
                use_checkpoint=True,
            )
            backbone.init_weights(args.teacher_pretrain_backbone)
        elif backbone == "swin_large":
            backbone = SwinTransformer(
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=7,
                ape=False,
                drop_path_rate=0.2,
                patch_norm=True,
                use_checkpoint=True,
            )
            backbone.init_weights(args.teacher_pretrain_backbone)
        elif backbone == "swin_large_window12":
            backbone = SwinTransformer(
                pretrain_img_size=384,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                ape=False,
                drop_path_rate=0.2,
                patch_norm=True,
                use_checkpoint=True,
            )
            backbone.init_weights(args.teacher_pretrain_backbone)
        else:
            raise NotImplementedError

        # for name, parameter in backbone.named_parameters():
        #     # TODO: freeze some layers?
        #     if not train_backbone:
        #         parameter.requires_grad_(False)

        self.body = backbone

    def forward(self, x):
        out = self.body(x)
        return out


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))
    def forward(self,x):
        return x + self.dummy - self.dummy #(also tried x+self.dummy)


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks: ExtraFPNBlock
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self,
                 backbone: nn.Module,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels = out_channels
        self.dummy_layer = DummyLayer()

    def forward(self, x):
        x = self.dummy_layer(x)
        x = self.body(x)
        x = self.fpn(x)
        return x


def build_backbone(args):
    if "resnet50" in args.teacher_backbone:
        train_backbone = args.lr_backbone > 0
        return_interm_layers = True
        resnet_backbone = Backbone(args.teacher_backbone, train_backbone, return_interm_layers, args.dilation)

        returned_layers = [1, 2, 3, 4]
        assert min(returned_layers) > 0 and max(returned_layers) < 5

        in_channels_stage2 = 256
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        out_channels = 256
        backbone = BackboneWithFPN(resnet_backbone, in_channels_list, out_channels)
    else:
        swin_backbone = SwinBackbone(args.teacher_backbone, args)
        out_channels = 256
        if "swin_tiny" in args.teacher_backbone:
            in_channels_list = [96, 192, 384, 768]
        elif "swin_small" in args.teacher_backbone:
            in_channels_list = [96, 192, 384, 768]
        else:
            in_channels_list = [192, 384, 768, 1536]
        backbone = BackboneWithFPN(swin_backbone, in_channels_list, out_channels)

    return backbone
