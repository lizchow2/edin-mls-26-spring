import pytest
import torch
from glm_asr_triton_template.layers import layernorm_kernel

def triton_layernorm(x, w, b, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_kernel[grid](
        x, w, b, y,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

@pytest.mark.parametrize("batch_size, hidden_size", [
    (1, 512),
    (8, 1024),
    (4, 127),
    (2, 4096), 
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_layernorm(batch_size, hidden_size, dtype):
    torch.manual_seed(42)
    
    x = torch.randn((batch_size, hidden_size), device='cuda', dtype=dtype)
    w = torch.randn((hidden_size,), device='cuda', dtype=dtype)
    b = torch.randn((hidden_size,), device='cuda', dtype=dtype)
    eps = 1e-5

    tt_y = triton_layernorm(x, w, b, eps)
    pt_y = torch.nn.functional.layer_norm(x, (hidden_size,), w, b, eps)
    
    atol = 1e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(tt_y, pt_y, atol=atol, rtol=1e-4)