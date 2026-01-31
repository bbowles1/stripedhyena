#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 09:25:59 2026

@author: bbowles
"""

import torch
import torch.nn as nn

class PyTorchFFTConv(nn.Module):
    """
    Pure PyTorch FFT convolution - drop-in replacement for FlashFFTConv.
    Always computes in float32 for numerical stability, then converts to target dtype.
    """
    def __init__(self, fft_size, dtype=torch.float32):
        super().__init__()
        self.fft_size = fft_size
        self.dtype = dtype
    
    def forward(self, u, k):
        """
        Compute FFT-based convolution.
        
        Args:
            u: input tensor, shape (B, H, L)
            k: kernel tensor, shape (H, L) or (B, H, L)
        
        Returns:
            y: convolution output, shape (B, H, L)
        """
        # Store original dtype
        original_dtype = u.dtype
        
        # CRITICAL: Always compute in float32 for numerical stability
        # FFT operations are very sensitive to precision
        u_f32 = u.to(torch.float32)
        k_f32 = k.to(torch.float32)
        
        # Ensure kernel has batch dimension
        if k_f32.dim() == 2:
            k_f32 = k_f32.unsqueeze(0)
        
        # Broadcast kernel to match batch size if needed
        if k_f32.shape[0] == 1 and u_f32.shape[0] > 1:
            k_f32 = k_f32.expand(u_f32.shape[0], -1, -1)
        
        # Use the provided FFT size
        fft_s = self.fft_size
        
        # Compute FFT - using the same normalization as ref_fftconv
        u_fft = torch.fft.rfft(u_f32, n=fft_s, dim=-1)
        k_fft = torch.fft.rfft(k_f32, n=fft_s, dim=-1)
        
        # Pointwise multiply in frequency domain
        # Divide by fft_s to match ref_fftconv normalization
        y_fft = u_fft * k_fft / fft_s
        
        # Inverse FFT with 'forward' normalization (matches ref_fftconv)
        y = torch.fft.irfft(y_fft, n=fft_s, dim=-1, norm="forward")
        
        # Truncate to original sequence length
        y = y[..., :u.shape[-1]]
        
        # Convert to target dtype at the very end
        # This matches the behavior where computation is in float32
        # but output can be float16
        return y.to(self.dtype)
    

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as in_channels for depthwise
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=in_channels,  # Key parameter for depthwise convolution
            bias=bias
        )
    
    def forward(self, x):
        return self.conv(x)
    