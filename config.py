from models import BIT_TYPE_DICT


class Config:

    def __init__(self, bit_type=8, quant_method='minmax', ptf=False, lis=False):
        self.BIT_TYPE_W = BIT_TYPE_DICT[f'int{bit_type}']
        self.BIT_TYPE_A = BIT_TYPE_DICT[f'uint{bit_type}']

        self.BIT_TYPE_W_attn = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint8']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = 'omse'

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        self.CALIBRATION_MODE_A_v = 'channel_wise'
        self.CALIBRATION_MODE_S_v = 'channel_wise'

        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
