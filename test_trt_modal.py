import modal
import os

app = modal.App("onnx-to-tensorRT-app")



Image = (
    # modal.Image.debian_slim(python_version="3.10")
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.06-py3")
    # modal.Image.from_registry("nvcr.io/nvidia/tensorrt:21.12-py3")
    
    .apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgl1",
        "libfontconfig1",
    )
    .pip_install(
        "fastapi==0.85.0",
        "onnx",
        "onnxruntime-gpu==1.12.1",
        "onnxsim==0.4.8",
        "wheel",
        "setuptools",
        "packaging",   
        "requests",
        "logzero",
        "pillow",
        "numpy==1.24.4",
        "pyyaml",
        "torch==1.11.0",
        "torchvision==0.12.0",
        "gdown",
        "einops"

    )
)



# ---- MAIN FUNCTION ----
@app.function(gpu="T4", image=Image)
def test_trt():
    import subprocess
    import requests
    import torch
    import tensorrt as trt
    from logzero import logger
    import sys



    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 1️⃣ Kiểm tra GPU và driver
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version:" in output
    assert "CUDA Version:" in output

    print("=========== GPU INFO ===========")
    print(output)
    print("TensorRT version:", trt.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0))

    # 2️⃣ In danh sách API có trong TensorRT (để debug)
    print("=========== TensorRT API ===========")
    print([x for x in dir(trt) if "Calib" in x or "Builder" in x])

    import os
    os.system("nvidia-smi")
    print("TensorRT version:", trt.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    # --- Bước 1: Clone repo vào thư mục con ---
    repo_url = "https://github.com/Kin-afk-ops/test_ocr_trt_tran.git"
    repo_dir = "test_ocr_trt_tran"

    if not os.path.exists(repo_dir):
        cmd_clone = f"git clone {repo_url} {repo_dir}"
        result = subprocess.run(cmd_clone, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"❌ Git clone failed with code {result.returncode}")
    else:
        print(f"📁 Repository already exists at {repo_dir}, pulling latest changes...")
        cmd_pull = f"git -C {repo_dir} pull"
        result = subprocess.run(cmd_pull, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"❌ Git pull failed with code {result.returncode}")

    # --- Bước 2: Chuyển vào thư mục repo ---
    os.chdir(repo_dir)
    print(f"📂 Current directory: {os.getcwd()}")

   
 

     # --- Bước 3: Chạy run.py trong repo ---
    run_py = "/workspace/test_ocr_trt_tran/trt_ocr_demo.py"
    if os.path.exists(run_py):
        print(f"▶️ Running {run_py} ...")
        result = subprocess.run(
            f"{sys.executable} {run_py}",
            shell=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"❌ /workspace/test_ocr_trt_tran/trt_ocr_demo.py failed with code {result.returncode}")
    else:
        raise FileNotFoundError(f"❌ {run_py} not found in repo.")



