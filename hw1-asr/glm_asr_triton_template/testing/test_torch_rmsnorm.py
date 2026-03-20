import torch
import triton
import triton.testing

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['hidden_size'], 
        x_vals=[512, 1024, 2048, 4096, 8192], 
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch (Native)', 'Triton (Fused)'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='rmsnorm-performance',
        args={'batch_size': 32} # Fixed batch size for comparison
    )
)
def benchmark(batch_size, hidden_size, provider):
    x = torch.randn((batch_size, hidden_size), device='cuda', dtype=torch.float32)
    weight = torch.ones(hidden_size, device='cuda', dtype=torch.float32)
    eps = 1e-6

    if provider == 'torch':
        # Using a simple fused eager implementation for baseline
        def torch_rmsnorm(x, w, eps):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        return triton.testing.do_bench(lambda: torch_rmsnorm(x, weight, eps))

    if provider == 'triton':
        return triton.testing.do_bench(lambda: run_rmsnorm(x, weight, eps))

# Assuming run_rmsnorm and rmsnorm_kernel are defined
benchmark.run(show_plots=True, print_data=True)