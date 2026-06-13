# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

# to learn to be
class BaseObserver:

    def __init__(self, module_type, bit_type, calibration_mode):
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.max_val = None
        self.min_val = None
        self.eps = torch.finfo(torch.float32).eps

    def reshape_tensor(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        v = v.detach()
        if self.module_type in ['conv_weight', 'linear_weight']:
            # BCHW -> B(CHW)
            v = v.reshape(v.shape[0], -1)
        elif self.module_type == 'activation':
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1) # [B, width, grid, grid] -> [B, grid, grid, width] 
            v = v.reshape(-1, v.shape[-1]) #  dim=4: [B, grid, grid, width] ->  [B * grid * grid, width]; dim=3: [grid*grid + 1, B, width] ->  [grid*grid + 1 * B, width]
            v = v.transpose(0, 1)  # [B * grid * grid, width]  -> [width, B * grid * grid] ; dim=3: [(grid*grid + 1) * B, width] -> [width, (grid*grid + 1) * B]
        else:
            raise NotImplementedError
        return v

    def update(self, v):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        # returns scale, zero_point
        raise NotImplementedError
