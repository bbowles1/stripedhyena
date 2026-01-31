import argparse

import pytest
import torch
import torch.nn as nn
from stripedhyena.layers import RMSNorm
from stripedhyena.utils import dotdict


def test_aa_fp_error(pytestconfig):
    """Test floating-point consistency between batch sizes"""
    input_dim = 1000
    output_dim = 1000
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = nn.Linear(input_dim, output_dim).to(device).to(dtype)

    # Create input on the correct device
    x1 = torch.randn(1, input_dim, dtype=dtype, device=device)
    x4 = x1.repeat(4, 1)

    y1 = linear(x1)
    y4 = linear(x4)

    if pytestconfig.getoption("verbose") > 0:
        print("y1[0]:", y1[0])
        print("y4[0]:", y4[0])
        diff = (y1[0] - y4[0]).abs()
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")

    # Test that repeated input gives same output
    assert torch.allclose(y1[0], y4[0], rtol=1e-2, atol=1e-3), \
        "Linear layer output differs between batch sizes"


def test_batched_norm(pytestconfig):
    """Test RMSNorm consistency between batch sizes"""
    config = {
        "eps": 1e-5,
        "hidden_size": 64,
        "params_dtype": torch.float32,
        "use_flash_rmsnorm": False,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dotdict(config)
    rmsnorm = RMSNorm(config).to(device).to(torch.bfloat16)

    # Create input with correct shape (batch, seq_len, hidden_size)
    inputs = torch.randn(1, 1, 64, dtype=torch.bfloat16, device=device)
    inputs = inputs.repeat(4, 1, 1)
    
    outputs_1 = rmsnorm(inputs[:1])
    outputs_4 = rmsnorm(inputs)

    if pytestconfig.getoption("verbose") > 0:
        print("outputs_1:", outputs_1)
        print("outputs_4[0]:", outputs_4[0])
        diff = (outputs_1[0] - outputs_4[0]).abs()
        print(f"Max difference: {diff.max()}")
        print(f"Mean difference: {diff.mean()}")

    # Test that normalization is consistent across batch sizes
    assert torch.allclose(outputs_1[0], outputs_4[0], rtol=1e-2, atol=1e-3), \
        "RMSNorm output differs between batch sizes"