import modal
import shutil, os

volume = modal.Volume.from_name("asr-benchmark-results", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", 
        add_python="3.11"
    )
    .apt_install("git", "libsndfile1", "ffmpeg", "nsight-systems-12.4")
    .pip_install(
        "numpy<2.0",
        "torch==2.10.0",
        "triton==3.6.0",
        "transformers",
        "huggingface_hub",
        "accelerate",
        "datasets<2.20.0",
        "scipy",
        "pysoundfile",
    )
    # This ensures your MSc AI weights are ready immediately
    .run_commands(
        "python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(\"zai-org/GLM-ASR-Nano-2512\")'"
    )
    .add_local_dir("./hw1-asr", remote_path="/root/hw1-asr", copy=True)
    .run_commands("chmod +x /root/hw1-asr/benchmark.sh")
)

app = modal.App("mls-triton-pro", image=image)

@app.function(
    gpu="H200",
    image=image,
    volumes={"/root/results": volume},
    timeout=3600,
)
def run_triton_benchmark(cmd: str):
    import subprocess

    cmd_breakdown = cmd.split()
    benchmark_sh = cmd_breakdown.pop(0)
    folder_name = cmd_breakdown.pop(0)
    try:
        add_cmd = cmd_breakdown.pop(0)
    except IndexError:
        add_cmd = None

    cwd = "/root/hw1-asr"
    log_path = f"/root/results/{folder_name}_benchmark.log"

    print(f"--- Starting Triton Benchmark: {folder_name} ---")
    
    process = subprocess.Popen(
        f"stdbuf -oL -eL {cmd} 2>&1 | tee {log_path}",
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        print(line, end="")

    if add_cmd == "--nsys":
        shutil.copy2(f"{cwd}/profile_{folder_name}.nsys-rep", f"/root/results/profile_{folder_name}.nsys-rep")

    process.wait()
    volume.commit()
    return process.returncode

@app.local_entrypoint()
def main():
    run_triton_benchmark.remote(
        cmd = f"./benchmark.sh glm_asr_triton_example"
    )