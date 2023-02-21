import torch
from torch import nn


class Conv3DBlock(nn.Module):
    """Conv3D + LeakyReLU(negative_slope=0.1)"""
    def __init__(
        self,
        channel_in: int, 
        channel_out: int,
        kernel: int = 3,
        stride: int = 1,
        padding: str = 'same'
    ):
        super().__init__()

        self._layer = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, kernel, stride, padding),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        return self._layer(input)

class PlainCNN(nn.Module):
    """
        12 Conv3DBlock + Global Residual

        This model only accepts grayscale images as input.
    """
    def __init__(
        self,
        channel_in: int = 1,
        channel_out: int = 1,
        channel_middle: int = 64,
        block_number: int = 12,
    ):
        super(PlainCNN, self).__init__()

        self._block_in = Conv3DBlock(channel_in, channel_middle)
        self._block_middle_list = nn.ModuleList(
            Conv3DBlock(channel_middle, channel_middle)
            for i in range(block_number-2)
        )
        self._block_out = Conv3DBlock(channel_middle, channel_out)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.clone()

        for block in [self._block_in, *self._block_middle_list, self._block_out]:
            output = block(output)
        
        return input + output
