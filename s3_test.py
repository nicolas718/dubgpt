import boto3
import os

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

def upload_file_to_s3(local_path, s3_key):
    s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
    print(f"Uploaded {local_path} to s3://{S3_BUCKET_NAME}/{s3_key}")

def download_file_from_s3(s3_key, local_path):
    s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
    print(f"Downloaded s3://{S3_BUCKET_NAME}/{s3_key} to {local_path}")

if __name__ == "__main__":
    # Create a test file
    with open("test_upload.txt", "w") as f:
        f.write("hello from polydub!")

    # Upload
    upload_file_to_s3("test_upload.txt", "test/test_upload.txt")

    # Download
    download_file_from_s3("test/test_upload.txt", "test_downloaded.txt")
