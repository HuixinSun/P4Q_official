# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer

from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
    
class QConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type) 

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight) # self.weight initialized in nn.Conv2d
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer(self.weight) # quantizer forward
        # x: float, weight: float -> int -> float = output: float
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
        
    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight) # quantizer forward, nn.Linear weight & bias
        return F.linear(x, weight, self.bias)


class QAct(nn.Module):
    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
        # self.vis_value = None
        # self.ori_value = None
        
    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(x) 
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x) 
        if not self.quant:
            return x
        
        # self.ori_value = x
        x = self.quantizer(x)
        # self.vis_value = x
        # x: float -> int -> float
        return x

# todo: check details
class QIntLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps,
                                            elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 7
        N = torch.clamp(bit - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2**(bit + 1) - 1)
        return M, N

    def forward(self,
                x,
                in_quantizer=None,
                out_quantizer=None,
                in_scale_expand=1):
        
        if self.mode == 'ln':
             # merge with clip
            orig_type = x.dtype
            x = x.type(torch.float32)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            return x.type(orig_type)
        elif self.mode == 'int':
            # input fq16, output fq 16
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(
                    -1, in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            assert in_scale is not None and out_scale is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round() # quantize
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()

            x_q = x_q * in_scale_mask

            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(
                channel_nums * (x_q**2).sum(dim=-1) - x_q.sum(dim=-1)**2)

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) -
                  (mean_x_q / std_x_q).unsqueeze(-1) *
                  self.weight.reshape(1, 1, -1)) / out_scale *
                 torch.pow(2, N)).round()

            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            x = x_q * out_scale
            
            return x
        else:
            raise NotImplementedError

# todo: check details
class QIntSoftmax(nn.Module):

    def __init__(self,
                 log_i_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor

        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = x.softmax(dim=-1)  # post-softmax activation
            if self.calibrate:  
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x


class QMultiheadAttention(nn.MultiheadAttention):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, 
                quant=False,
                calibrate=False,
                last_calibrate=False,
                bit_type=BIT_TYPE_DICT["int8"],
                calibration_mode="channel_wise",  # weight
                observer_str="minmax",
                quantizer_str="uniform", 
                bit_type_a=BIT_TYPE_DICT["int8"], # activation
                calibration_mode_a="channel_wise", # channel_wise
                observer_str_a="minmax",
                quantizer_str_a="uniform"):
        super(QMultiheadAttention, self).__init__(embed_dim=embed_dim, num_heads=num_heads)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        
        self.bit_type_a = bit_type_a
        self.calibration_mode_a = calibration_mode_a
        self.observer_str_a = observer_str_a
        self.quantizer_str_a = quantizer_str_a
        
        self.module_type = "linear_weight"
        
        # quantize enabled
        self.observer_in = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_out = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_q = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_k = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_v = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)

        self.quantizer_in = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_in, self.module_type)
        self.quantizer_out = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_out, self.module_type)
        self.quantizer_q = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_q, self.module_type)
        self.quantizer_k = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_k, self.module_type)
        self.quantizer_v = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_v, self.module_type)
        # self.vis_in_proj = None
        # self.original_in_proj = None
        # self.vis_out_proj = None
        # self.original_out_proj = None
        # self.vis_q = None
        # self.original_q = None

        # if torch.equal(query, key) and torch.equal(key, value):
        #     # self-attention
        #     q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
            
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):

        if self.calibrate:
            if self._qkv_same_embed_dim:
                self.quantizer_in.observer.update(self.in_proj_weight) # self.in_proj_weight in nn.MultiheadAttention
                self.quantizer_out.observer.update(self.out_proj.weight)
                if self.last_calibrate:
                    self.quantizer_in.update_quantization_params(query)
                    self.quantizer_out.update_quantization_params(query)
            else:
                self.quantizer_q.observer.update(self.q_proj_weight)
                self.quantizer_k.observer.update(self.k_proj_weight)
                self.quantizer_v.observer.update(self.v_proj_weight)

                if self.last_calibrate:
                    self.quantizer_in.update_quantization_params(query)
                    self.quantizer_out.update_quantization_params(query)
                    self.quantizer_q.update_quantization_params(query)
                    self.quantizer_k.update_quantization_params(key)
                    self.quantizer_v.update_quantization_params(value)
        
        if not self._qkv_same_embed_dim:
            if not self.quant:
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight)
            else:
                in_proj_weight = self.quantizer_in(self.in_proj_weight)
                out_proj_weight = self.quantizer_out(self.out_proj.weight)
                q_proj_weight = self.quantizer_q(self.q_proj_weight) 
                k_proj_weight = self.quantizer_k(self.k_proj_weight)
                v_proj_weight = self.quantizer_v(self.v_proj_weight)
                
                
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, out_proj_weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight)

        else:
            if not self.quant:
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)
            else:
                in_proj_weight = self.quantizer_in(self.in_proj_weight)
                out_proj_weight = self.quantizer_out(self.out_proj.weight)

                # self.vis_in_proj = in_proj_weight
                # self.vis_out_proj = out_proj_weight
                
                # self.original_in_proj = self.in_proj_weight
                # self.original_out_proj = self.out_proj.weight

                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, out_proj_weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)
    