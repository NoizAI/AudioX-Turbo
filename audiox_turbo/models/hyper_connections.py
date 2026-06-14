import torch
from torch import nn
from typing import Callable, Tuple

class Residual(nn.Module):
    """简单的残差连接类"""
    def __init__(self, num_streams: int = 1, dim: int = None):
        super().__init__()
        self.num_streams = num_streams
        
    def forward(self, x):
        if self.num_streams == 1:
            return x, lambda tokens: tokens
        else:
            # 对于多流情况，返回输入和恒等函数
            return x, lambda tokens: tokens

class HyperConnections(nn.Module):
    """超连接类，用于处理多流残差连接"""
    def __init__(self, num_streams: int, dim: int):
        super().__init__()
        self.num_streams = num_streams
        self.dim = dim
        
        if num_streams > 1:
            # 创建扩展和减少流的线性层
            self.expand = nn.Linear(dim, dim * num_streams)
            self.reduce = nn.Linear(dim * num_streams, dim)
    
    def forward(self, x):
        if self.num_streams == 1:
            return x, lambda tokens: tokens
        
        # 扩展流
        expanded = self.expand(x)
        expanded = expanded.view(*x.shape[:-1], self.num_streams, self.dim)
        
        # 返回扩展后的张量和减少函数
        return expanded, lambda tokens: self.reduce(tokens.view(*tokens.shape[:-2], -1))
    
    @staticmethod
    def get_expand_reduce_stream_functions(num_streams: int, disable: bool = False):
        """获取扩展和减少流的函数"""
        if disable or num_streams == 1:
            # 返回恒等函数
            return lambda x: x, lambda x: x
        
        def expand_streams(x):
            # 扩展流
            if x.dim() == 3:  # [batch, seq, dim]
                return x.unsqueeze(2).expand(-1, -1, num_streams, -1)
            elif x.dim() == 4:  # [batch, seq, streams, dim]
                return x
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        def reduce_streams(x):
            # 减少流
            if x.dim() == 4:  # [batch, seq, streams, dim]
                return x.mean(dim=2)  # 平均所有流
            elif x.dim() == 3:  # [batch, seq, dim]
                return x
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        return expand_streams, reduce_streams












