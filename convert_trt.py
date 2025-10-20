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
    from logzero import logger



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

       # 1️⃣ Download ONNX
    onnx_path = "/tmp/transformer_decoder.onnx"
    print(f"Downloading ONNX model from {onnx_url}")
    r = requests.get(onnx_url)
    with open(onnx_path, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved ONNX to {onnx_path}")
    

    onnx_data_path = "/tmp/transformer_encoder.onnx.data"
    print(f"Downloading ONNX model from {onnx_data_url}")
    r = requests.get(onnx_data_url)
    with open(onnx_data_path, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved ONNX to {onnx_data_path}")

    subprocess.run("ls /usr/src/tensorrt/bin", shell=True)

    
    os.makedirs("/tmp", exist_ok=True)
    # Lệnh theo tài liệu NVIDIA Quick Start
    engine_path = f"/tmp/{output_engine_name}"


    # cmd1 =  [
    #     "python -m onnxsim /tmp/model.onnx /tmp/new_model.onnx --overwrite-input-shape=5,3,32,160"
    #     # "python -m onnxsim /tmp/model.onnx /tmp/new_model.onnx --dynamic-input-shape --input-shape tgt_inp:20,1 memory:170,1,256"
    # ]
    # result1 = subprocess.run(cmd1, capture_output=True, text=True, shell=True)
    # print(result1.stdout)
    # print(result1.stderr)
    
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx=/tmp/transformer_decoder.onnx",
        f"--saveEngine={engine_path}",
        # "--minShapes=input:1x3x32x128",
        # "--optShapes=input:32x3x32x512",
        # "--maxShapes=input:32x3x32x768",
        "--minShapes=tgt_inp:1x1,memory:64x1x256",
        "--optShapes=tgt_inp:64x32,memory:256x32x256",
        "--maxShapes=tgt_inp:128x32,memory:384x32x256"
        "--verbose",
        "--fp16"
    ]

    # Thực thi lệnh
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"❌ Conversion failed with code {result.returncode}")

    print(f"✅ TensorRT engine saved at {engine_path}")

     # 4️⃣ Trả engine về client
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
