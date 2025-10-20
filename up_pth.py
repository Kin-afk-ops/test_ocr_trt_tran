import boto3
from botocore.client import Config

R2_ACCESS_KEY = "ad0b62e83097180a3800247094003f65"
R2_SECRET_KEY = "8aa1992e91490330824747d4e25e40a170fb0877a4bbb996c8c6e703531ec7bd"
R2_ENDPOINT_URL = "https://a1000a80a775f57fe92ea14196486a3a.r2.cloudflarestorage.com"
R2_BUCKET_NAME = "ai-detect"
R2_PUBLIC_URL = "https://r2-storage.teknix.services"


def upload_to_r2(local_path, r2_key):
    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        endpoint_url=R2_ENDPOINT_URL,
        config=Config(signature_version="s3v4"),
        region_name="auto"
    )
    with open(local_path, "rb") as f:
        client.upload_fileobj(f, R2_BUCKET_NAME, r2_key)
    print(f"Uploaded: {local_path} → {R2_PUBLIC_URL}/{r2_key}")


# Upload file .pth:
if __name__ == "__main__":
    # local_file = r"D:\code\thuc_tap_2025\teknix\15_10_2025\modal_ocr\weight\seq\decoder.onnx"
    local_file = r"D:\code\thuc_tap_2025\teknix\20_10_2025\transformer_decoder.trt"
    # local_file = r"D:\code\thuc_tap_2025\teknix\14_10_2025\seq_train\config_seq_14_10_2025.yml"
    r2_object_key = "models/vietocr/modal_ocr/onnx/tran/new/transformer_decoder.trt"
    upload_to_r2(local_file, r2_object_key)
