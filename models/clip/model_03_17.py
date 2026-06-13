from collections import OrderedDict
from typing import Tuple, Union
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..layers_quant import DropPath, HybridEmbed, Mlp, PatchEmbed, trunc_normal_
from ..ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear

import clip

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        # self.qkv
        # reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
        self.in_proj = QLinear(dim,
                           dim * 3,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        # self.proj
        self.out_proj = QLinear(dim,
                            dim,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact_attn1 = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.log_int_softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S)

    def forward(self, x):
        B, N, C = x.shape
        x = self.in_proj(x)
        x = self.qact1(x)
        qkv = x.reshape(B, N, 3, self.num_heads,
                        C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.qact_attn1(attn)
        attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.qact2(x)
        x = self.out_proj(x)
        x = self.qact3(x)
        x = self.proj_drop(x)
        return x
    
# to enable quant
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # quant enabled
        self.conv1 = QConv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = QConv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = QConv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

# to enable quant
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, 
                qkv_bias=False, quant=False, calibrate=False, cfg=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)  # no quant
        # self.k_proj = nn.Linear(embed_dim, embed_dim)
        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.v_proj = nn.Linear(embed_dim, embed_dim)
        # self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.k_proj = QLinear(embed_dim,
                           embed_dim,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.q_proj = QLinear(embed_dim,
                           embed_dim,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.v_proj = QLinear(embed_dim,
                           embed_dim,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.c_proj = QLinear(embed_dim,
                           output_dim or embed_dim,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        
    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # to modify with Attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

# to enable quant  
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem: quant enabled
        self.conv1 = QConv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2) # to add
        self.relu1 = nn.ReLU(inplace=True) # to add
        self.conv2 = QConv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = QConv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

# merged with QIntLayerNorm
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

# to enable quant 
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 # quant
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 ln_attn=None,
                 act_layer=None,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        # merge with fq-vit: add activation
        # self.ln_1 = LayerNorm(d_model) 
        ln_attn = ln_attn or partial(nn.LayerNorm, eps=1e-6)
        self.ln_1 = ln_attn(d_model)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        
        # merge with fq-vit: substitute and add activation
        # self.attn = nn.MultiheadAttention(d_model, n_head) 
        self.attn = Attention(d_model,
                        num_heads=n_head,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        cfg=cfg)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        
        # merge with fq-vit: add activation
        # self.ln_2 = LayerNorm(d_model)
        self.ln_2 = ln_attn(d_model)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
    
        # merge with fq-vit: substitute and add activation
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        self.mlp = Mlp(in_features=d_model,
                hidden_features=d_model * 4,
                act_layer=act_layer,
                quant=quant,
                calibrate=calibrate,
                cfg=cfg)
        self.qact4 = QAct(quant=quant,
                    calibrate=calibrate,
                    bit_type=cfg.BIT_TYPE_A,
                    calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                    observer_str=cfg.OBSERVER_A_LN,
                    quantizer_str=cfg.QUANTIZER_A_LN)
        
        # not used here
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor,last_quantizer=None):
        # merge with fq-vit: omit drop-path
        # x = x + self.attention(self.ln_1(x))
        x = x + self.qact2(self.attn(self.qact1(self.ln_1(x, last_quantizer, self.qact1.quantizer))))
        
        # merge with fq-vit: omit drop-path
        # x = x + self.mlp(self.ln_2(x))
        x = self.qact4(x + 
                self.mlp(
                    self.qact3( 
                        self.ln_2(x, self.qact2.quantizer,
                                self.qact3.quantizer))))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                ln_attn=None,
                quant=False,
                calibrate=False,
                cfg=None):
        super().__init__()
        self.width = width
        self.layers = layers
        ln_attn = ln_attn or partial(nn.LayerNorm, eps=1e-6)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, qkv_bias=True,act_layer=QuickGELU, ln_attn=ln_attn, quant=quant, calibrate=calibrate,cfg=cfg) for i in range(layers)]) 

    def forward(self, x: torch.Tensor):
        for i, blk in enumerate(self.resblocks):
            last_quantizer = blk.qact1.quantizer if i == 0 else self.resblocks[
                i - 1].qact4.quantizer
            x = blk(x, last_quantizer)
        # return self.resblocks(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 # quant
                 qkv_bias=True,
                 ln_pre=None,
                 ln_attn=None,
                 ln_post=None,
                 quant=False,
                 calibrate=False,
                 input_quant=False,
                 cfg=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        
        self.cfg = cfg
        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
        # patch emb: add activation
        self.conv1 = QConv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False) # quant enabled
        self.qact_conv1= QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        scale = width ** -0.5
        # cls emb: add activation
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) # different initilization
        # self.qact_cls = QAct(quant=quant,
        #                 calibrate=calibrate,
        #                 bit_type=cfg.BIT_TYPE_A,
        #                 calibration_mode=cfg.CALIBRATION_MODE_A,
        #                 observer_str=cfg.OBSERVER_A,
        #                 quantizer_str=cfg.QUANTIZER_A)
        # pos emb: add two activations
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)) # different initilization
        self.qact_pos_pre = QAct(quant=quant,
                        calibrate=calibrate,
                        bit_type=cfg.BIT_TYPE_A,
                        calibration_mode=cfg.CALIBRATION_MODE_A,
                        observer_str=cfg.OBSERVER_A,
                        quantizer_str=cfg.QUANTIZER_A)
        self.qact_pos_post = QAct(quant=quant,
                        calibrate=calibrate,
                        bit_type=cfg.BIT_TYPE_A,
                        calibration_mode=cfg.CALIBRATION_MODE_A,
                        observer_str=cfg.OBSERVER_A,
                        quantizer_str=cfg.QUANTIZER_A)
        # merge with fq-vit: 
        # self.ln_pre = nn.LayerNorm(width) ?
        if ln_pre:
            self.qact_before_ln_pre = QAct(
                quant=quant,
                calibrate=calibrate,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A)
            self.ln_pre  = ln_pre(width) 
            self.qact_ln_pre = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        else:
            self.qact_before_ln_pre = nn.Identity() # Identity() is empty operation
            self.ln_pre = nn.Identity()
            self.qact_ln_pre =  QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        
        # modify in Transformer
        self.transformer = Transformer(width, layers, heads,
                                    ln_attn=ln_attn,
                                    quant=quant,
                                    calibrate=calibrate,
                                    cfg=cfg)
        # merge with fq-vit: 
        # self.ln_post = LayerNorm(width)
        ln_post  = ln_post  or partial(nn.LayerNorm, eps=1e-6)
        if ln_post :
            self.qact_before_ln_post = QAct(
                quant=quant,
                calibrate=calibrate,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A)
            self.ln_post  = ln_post(width) # to read
            self.qact_ln_post = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        else:
            self.qact_before_ln_post  = nn.Identity() # Identity() is empty operation
            self.ln_post  = nn.Identity()
            self.qact_ln_post  =  QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        
        # proj: add activation
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim)) # Representation layer?
        self.qact_proj = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
       
    def forward(self, x: torch.Tensor):
        if self.input_quant:
            x = self.qact_input(x)
        
        # merge with fq-vit: add one activation
        # x = self.conv1(x) # shape = [*, width, grid, grid],  (conv1): Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32), bias=False), width=768
        x = self.conv1(x).flatten(2).transpose(1, 2) # [*, width=768, grid, grid] -> [*, width, grid ** 2] -> [*, grid ** 2, width], quant enabled
        x = self.qact_before_ln_pre(x)
        # delete in clip:  line 457 does this
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # merge with fq-vit: add activation
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat((cls_tokens, x), dim=1) # [*, grid ** 2, width] -> [*, grid ** 2 + 1, width]
        x = self.qact_cls(x)
        # print(x.max())
      
        # merge with fq-vit: add two activations
        # x = x + self.positional_embedding.to(x.dtype)
        x = x + self.qact_pos_pre(self.positional_embedding.to(x.dtype)) # [1, num_patches + 1, embed_dim] = [*, grid ** 2 + 1, width] 
        x = self.qact_pos_post(x)
        # print(x.max())
        
        # replace with fq-vit
        # x = self.ln_pre(x) # LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        if isinstance(self.ln_pre , nn.Identity):
            x = self.ln_pre(x)
        else:
            x = self.ln_pre(x, self.qact_pos_post.quantizer, self.qact_ln_pre.quantizer)
        x = self.qact_ln_pre(x) 
        # print(x.max())
        
        # no drop-out
        x = x.permute(1, 0, 2)  # NLD -> LND, [*, grid ** 2 + 1, width] -> [grid ** 2 + 1, *, width], batch-size, layer num, dim
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print(x.max())
        # x = self.ln_post(x[:, 0, :])
        if isinstance(self.ln_post, nn.Identity):
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 0, :], self.transformer.resblocks[-1].qact4.quantizer, self.qact_ln_post.quantizer)
        x = self.qact_ln_post(x) # add activation
        # print(x.max())
        if self.proj is not None:
            x = x @ self.qact_proj(self.proj) # [batch_size, out_dim]
        print(x)
        # no final activation
        return x

class QCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # quant
                 qkv_bias=False,
                 quant=False,
                 calibrate=False,
                 input_quant=False,
                 # input quant config
                 cfg=None):
        super().__init__()
        
        self.cfg = cfg
        self.context_length = context_length

        # visual
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                # quant
                qkv_bias=True,
                ln_pre=partial(QIntLayerNorm, eps=1e-6), # PatchEmbedding layer_norm
                ln_attn=partial(QIntLayerNorm, eps=1e-6),
                ln_post=partial(QIntLayerNorm, eps=1e-6),
                quant=quant,
                calibrate=calibrate,
                input_quant=False,
                cfg=cfg
            )

        # text
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            ln_attn=partial(QIntLayerNorm, eps=1e-6),
            quant=quant,
            calibrate=calibrate,
            cfg=cfg
        )

        # to add: text related
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)

            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False
   
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (QConv2d, QLinear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, Attention):
            for attr in [*[f"{s}_proj.weight" for s in ["in", "q", "k", "v"]], "in_proj.bias", "bias_k", "bias_v"]:
                if hasattr(l, attr):
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()
                    
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
                    
    model.apply(_convert_weights_to_fp16)

def build_model(state_dict: dict, cfg = None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = QCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        qkv_bias=True,
        quant=False,
        calibrate=False,
        input_quant=False,
        cfg=cfg
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    # modify clip state_dict
    rep_dict = {'.attn.in_proj_weight':'.attn.in_proj.weight','.attn.in_proj_bias':'.attn.in_proj.bias'}
    for prefix in ['visual.transformer.resblocks.','transformer.resblocks.']: 
        for postfix in rep_dict.keys():
            for i in range(transformer_layers):
                rep_key = prefix + str(i) + postfix
                rep_key_to = prefix + str(i) + rep_dict[postfix]
            
                rep_value = state_dict[rep_key]
                state_dict[rep_key_to] = rep_value
                del state_dict[rep_key]
                
    # add fp 16: state_dict dtype 16
    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
