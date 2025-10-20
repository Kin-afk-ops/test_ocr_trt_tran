import numpy as np 
from exec_backends.trt_loader import TrtOCREncoder
import onnxruntime as rt
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from torch import nn

class TorchEncoder(nn.Module):
    def __init__(self, model):
        
        super(TorchEncoder, self).__init__()
        
        self.model = model

    def forward(self, img):
        """
        Shape:
            - img: (N, C, H, W)
            - output: b t v
        """
        src = self.model.cnn(img)
        # print('CNN out', src.shape)
        memory = self.model.transformer.forward_encoder(src)

        # memories = self.model.transformer.forward_encoder(src)
        return memory

config = Cfg.load_config_from_file("./tran_config_14_10_2025.yml")

config['weights'] = "https://r2-storage.teknix.services/models/vietocr/modal_ocr/onnx/tran/new/transformerocr_14_10_2025_final.pth"
config['device'] = 'cuda:0'

trainer = Predictor(config)

sample_inp = np.array(np.random.rand(6, 3, 32, 160), dtype = np.double)

trt_model = TrtOCREncoder('transformer_encoder.trt')
torch_model = TorchEncoder(trainer.model)
torch_model.eval()
onnx_model = rt.InferenceSession('transformer_encoder.onnx')

trt_out = np.squeeze(trt_model.run(sample_inp.copy().astype('float32')))
with torch.no_grad():
    torch_out = np.squeeze(torch_model(torch.Tensor(sample_inp.copy())).detach().cpu().numpy())

onnx_inp = {onnx_model.get_inputs()[0].name: sample_inp.copy().astype('float32')} 
onnx_out = np.squeeze(onnx_model.run(None, onnx_inp))
print(torch_out[10: 20, 0, 0])
print(onnx_out[10: 20, 0, 0])
print(trt_out[10: 20, 0, 0])