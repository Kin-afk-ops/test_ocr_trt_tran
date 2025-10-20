from PIL import Image
import torch
from vietocr.tool.translate import process_input,build_model
from vietocr.tool.config import Cfg
from vietocr.model.vocab import Vocab
import onnxruntime
import numpy as np

# --------- Hàm load ảnh và convert sang tensor ---------
def image_processing(img_path, config_path,seq_model_path):
    # --- Kiểm tra ảnh ---
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Ảnh không tồn tại: {img_path}")

    # --- Load config ---
    config = Cfg.load_config_from_file(config_path)

    # --- Chọn device tự động ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    # --- Load model & vocab ---
    model,vocab = build_model(config)
    # load weight
    model.load_state_dict(torch.load(seq_model_path, map_location=torch.device(config['device'])))
    model = model.eval()

    # --- Load và xử lý ảnh ---
    img = Image.open(img_path)
    img = process_input(img, config['dataset']['image_height'],
                config['dataset']['image_min_width'], config['dataset']['image_max_width'])
    img = img.to(config['device'])
    
    return img,model,vocab
# --------- Hàm dịch ảnh với 2 file ONNX transformer ---------
def translate_transformer_onnx(img, encoder_path, decoder_path, vocab, max_seq_length=128,sos_token=1, eos_token=2):
    """
    img_tensor: B x C x H x W
    vocab: instance Vocab đã khớp với ONNX
    """
    encoder_session = onnxruntime.InferenceSession(encoder_path)
    decoder_session = onnxruntime.InferenceSession(decoder_path)

    # Chạy encoder
    encoder_input = {encoder_session.get_inputs()[0].name: img.cpu().numpy().astype(np.float32)}
    encoder_outputs = encoder_session.run(None, encoder_input) # [B, seq_len, hidden]

    while max_length <= max_seq_length and not all(
        np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
    ):
        tgt_inp = translated_sentence
        decoder_input = {decoder_session.get_inputs()[0].name: tgt_inp[-1], decoder_session.get_inputs()[1].name: hidden, decoder_session.get_inputs()[2].name: encoder_outputs}

        output, hidden, _ = decoder_session.run(None, decoder_input)
        output = np.expand_dims(output, axis=1)
        output = torch.Tensor(output)

        values, indices = torch.topk(output, 1)
        indices = indices[:, -1, 0]
        indices = indices.tolist()

        translated_sentence.append(indices)
        max_length += 1

        del output

    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence

# --------- Hàm decode token sang text ---------
def decode_output(translated_array, vocab):
    batch_out = []
    for seq in translated_array:
        tokens = [t for t in seq if t not in [vocab.sos, vocab.eos, vocab.pad]]
        text = vocab.decode(tokens)
        batch_out.append(text)
    return batch_out

# ======================== MAIN ========================
img_path = r"C:\Users\ACER\Pictures\Screenshots\Screenshot 2025-10-13 162721.png"
config_path = r"D:\code\thuc_tap_2025\teknix\14_10_2025\tran_train\tran_config_14_10_2025.yml"
encoder_path = r'D:\code\thuc_tap_2025\teknix\20_10_2025\transformer_encoder.onnx'
decoder_path = r'D:\code\thuc_tap_2025\teknix\20_10_2025\transformer_decoder.onnx'

# Load ảnh
img_tensor, config = image_to_tensor(img_path, config_path)

# Load vocab từ config
vocab = Vocab(config['vocab'])
print("Vocab size:", len(vocab))

# Dịch ảnh
translated_tokens = translate_transformer_onnx(img_tensor, encoder_path, decoder_path, vocab)

# Decode sang text
text_batch = decode_output(translated_tokens, vocab)
print("Kết quả:", text_batch[0])
