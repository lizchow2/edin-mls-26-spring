import torch
import triton
import triton.language as tl
from layers import rmsnorm_kernel, FusedResidualRMSNorm
import triton.testing

def rmsnorm_pytorch(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms * weight

def run_rmsnorm(x, weight, eps=1e-6):
    batch_size, hidden_size = x.shape
    y = torch.empty_like(x)
    grid = (batch_size,)
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    
    rmsnorm_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        y_ptr=y,
        stride_x=x.stride(0),
        stride_y=y.stride(0),
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

def test_rmsnorm():
    torch.manual_seed(42)
    B, H = 32, 512 
    dtype = torch.float32
    eps = 1e-5

    x = torch.randn((B, H), device='cuda', dtype=dtype)
    weight = torch.randn((H,), device='cuda', dtype=dtype)

    triton_output = run_rmsnorm(x, weight, eps)
    torch_output = rmsnorm_pytorch(x, weight, eps)

    try:
        triton.testing.assert_close(triton_output, torch_output, atol=1e-5, rtol=1e-5)
        print("✅ Unit Test Passed: Triton output matches PyTorch!")
    except Exception as e:
        print("❌ Unit Test Failed!")
        print(e)
    
def test_benchmark():
    torch.manual_seed(42)
    B, H = 32, 1024 
    dtype = torch.float32
    eps = 1e-5

    x = torch.randn((B, H), device='cuda', dtype=torch.float32)
    res = torch.randn((B, H), device='cuda', dtype=torch.float32)
    fused_op = FusedResidualRMSNorm(H)
    # Move the parameter manually since the class doesn't have .cuda()
    fused_op.weight = torch.nn.Parameter(fused_op.weight.cuda())
    
    def semi_fused_op(x, res):
        # This creates an intermediate allocation and a write to DRAM
        added = x + res 
        # This calls your triton rmsnorm (assuming you have a non-residual version)
        # Or just use your current kernel but pass a zeroed-out residual
        return run_rmsnorm(added, fused_op.weight, fused_op.eps)

    ms_semi = triton.testing.do_bench(lambda: semi_fused_op(x, res))

    ms = triton.testing.do_bench(lambda: fused_op(x, res))
    print(f"Semi-Fused (Add + Triton Norm): {ms_semi:.4f} ms")
    print(f"Fully Fused (Triton AddNorm):  {ms:.4f} ms")
    print(f"Fusion Gain: {((ms_semi - ms) / ms_semi) * 100:.1f}% faster")

if __name__ == "__main__":
    test_benchmark()
