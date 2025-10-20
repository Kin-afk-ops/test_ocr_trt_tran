import modal
import os

app = modal.App("onnx-to-tensorRT-app")



Image = (
    #  modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.06-py3")
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
        "torch",
        "torchvision",  
        "onnx",
        "onnxruntime-gpu==1.12.1",
        "onnxsim==0.4.8",
        "wheel",
        "setuptools",
        "packaging",
        # "tensorrt-cu12",    
        "requests",
        "logzero",

    )
)



# ---- MAIN FUNCTION ----
@app.function(gpu="T4", image=Image)
def convert_onnx_to_engine(onnx_url: str, onnx_data_url, output_engine_name: str = "model.trt"):
    import subprocess
    import requests
    import torch
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Check GPU
    output = subprocess.check_output(["nvidia-smi"], text=True)
    print("=========== GPU INFO ===========")
    print(output)
    print("TensorRT version:", trt.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Download ONNX
    onnx_path = "/tmp/transformer_decoder.onnx"
    print(f"Downloading ONNX from {onnx_url}")
    r = requests.get(onnx_url)
    with open(onnx_path, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved ONNX to {onnx_path}")

    onnx_data_path = "/tmp/transformer_encoder.onnx.data"
    print(f"Downloading ONNX data from {onnx_data_url}")
    r = requests.get(onnx_data_url)
    with open(onnx_data_path, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved ONNX data to {onnx_data_path}")

    import os
    os.makedirs("/tmp", exist_ok=True)
    engine_path = f"/tmp/{output_engine_name}"

    # Build TensorRT engine with proper optimization profiles
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        # Optimization profiles for decoder
        # Format: tensor_name:min_shape,opt_shape,max_shape
        "--minShapes=tgt_inp:1x1,memory:64x1x256",
        "--optShapes=tgt_inp:64x32,memory:256x32x256",
        "--maxShapes=tgt_inp:128x32,memory:384x32x256",
        "--verbose",
        "--fp16",
    ]

    print("Running TensorRT conversion...")
    print(" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed with code {result.returncode}\n{result.stderr}")

    print(f"✅ TensorRT engine saved at {engine_path}")

    # Read and return engine
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()

    return engine_bytes



# ---- EXAMPLE CALL (local test) ----
@app.local_entrypoint()
def main():


    filename = os.path.basename("https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/transformer_decoder.onnx")
    engine_name = filename.replace(".onnx", ".trt")
    # engine_bytes = convert_onnx_to_engine.remote(
    #     "https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/new/transformer_encoder.onnx", 
    #     "https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/new/transformer_encoder.onnx.data",
    #     engine_name)
    

    engine_bytes = convert_onnx_to_engine.remote(
        "https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/new/transformer_decoder.onnx", 
        "https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/new/transformer_encoder.onnx.data",
        engine_name)

    local_path = f"./{engine_name}"
    with open(local_path, "wb") as f:
        f.write(engine_bytes)
    print(f"💾 Saved {engine_name} to {local_path}")
