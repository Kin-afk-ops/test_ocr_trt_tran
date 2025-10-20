import numpy as np
import torch
from torch import nn
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from exec_backends.trt_loader import TrtOCREncoder
import requests

class TorchEncoder(nn.Module):
    def __init__(self, model):
        super(TorchEncoder, self).__init__()
        self.model = model

    def forward(self, img):
        src = self.model.cnn(img)
        memory = self.model.transformer.forward_encoder(src)
        return memory

# Load config và model
config = Cfg.load_config_from_file("./tran_config_14_10_2025.yml")
config['weights'] = "https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/new/transformerocr_14_10_2025_final.pth"
config['device'] = 'cuda:0'
trainer = Predictor(config)

# Tạo input float32
sample_inp = np.random.rand(6, 3, 32, 160).astype(np.float32)

# Download TensorRT engine
encoder_path = "/tmp/transformer_encoder.trt"
r = requests.get("https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/new/transformer_encoder.trt")
with open(encoder_path, "wb") as f:
    f.write(r.content)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load models
trt_model = TrtOCREncoder(encoder_path)
torch_model = TorchEncoder(trainer.model).to(device)
torch_model.eval()

# Chạy TensorRT
trt_out = np.squeeze(trt_model.run(sample_inp))

# Chạy PyTorch (đưa input lên GPU)
input_tensor = torch.Tensor(sample_inp).to(device)
with torch.no_grad():
    torch_out = np.squeeze(torch_model(input_tensor).detach().cpu().numpy())

print(torch_out[10:20, 0, 0])
print(trt_out[10:20, 0, 0])
