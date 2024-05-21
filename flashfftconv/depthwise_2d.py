# Copyright (c) 2023, Dan Fu and Hermann Kumbong.
import torch
import math
from monarch_cuda import conv2d_forward
from einops import rearrange

class conv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, padding):
        outputs = conv2d_forward(input, weights, padding)
        ctx.padding = padding
        ctx.save_for_backward(input, weights)
        return outputs

    @staticmethod
    def backward(ctx, dout):
        input, weight = ctx.saved_tensors
        # dout  = dout.contiguous()
        # du, dk, dbias = conv1d_backward(dout, input, weight, bias, ctx.padding, ctx.is_bhl)
        # return du, dk, dbias, None, None
        return None

#TODO: initialization    
class FlashDepthWiseConv2d(torch.nn.Module):
    def __init__(self, channels, kernel_size, padding, weights, bias, is_bhl=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FlashDepthWiseConv2d, self).__init__()
        self.d = channels
        self.k = kernel_size
        self.padding = padding
        self.is_bhl = is_bhl
        if is_bhl:
            self.weights  = torch.nn.Parameter(weights.squeeze())
        else:
            self.weights  = torch.nn.Parameter(rearrange(weights.squeeze(), 'd k -> k d').detach().clone().contiguous())
        self.bias = torch.nn.Parameter(bias.detach().clone().contiguous())
        self.reset_parameters(weights, bias)

    #TODO: initialization
    def reset_parameters(self, weights, bias):
        pass
        # stdv = 1.0 / math.sqrt(self.state_size)
        # for weight in self.parameters():
        #     weight.data.uniform_(-stdv, +stdv)
    
    #current format for the weights is transpose of what is used in nn.Conv1d
    #[HK]: load the weights for the conv1d layer and then transpose them
    def load_state_dict(self, state_dict, strict: bool = True):
        pass
    
    #[HK]: transpose the weights before saving so that they can be loaded in nn.Conv1d
    def save_state_dict(self):
        pass
    
    def forward(self, input):
        return conv2dFunc.apply(input, self.weights, self.padding)