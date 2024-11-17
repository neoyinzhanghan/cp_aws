import os
import time
from dotenv import load_dotenv
from LLRunner.slide_processing.dzsave_h5 import dzsave_h5
from compute_heatmap import create_heatmap_to_h5
from tqdm import tqdm
from datetime import datetime, timedelta
import boto3

# Load environment variables from .env file
load_dotenv()

# S3 bucket and paths
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
s3_log_file_key = "put_log/s3_put_log.txt"  # Key for the PUT log file in S3
s3_subfolder = "wsi-and-heatmaps"

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

# Function to download the PUT log file from S3
def download_put_log():
    try:
        response = s3.get_object(Bucket=s3_bucket_name, Key=s3_log_file_key)
        return response["Body"].read().decode("utf-8").splitlines()
    except s3.exceptions.NoSuchKey:
        return []

# Function to upload the updated PUT log back to S3
def upload_put_log(logs):
    log_data = "\n".join(logs)
    s3.put_object(Bucket=s3_bucket_name, Key=s3_log_file_key, Body=log_data)

# Function to log a PUT request
def log_put_request():
    logs = download_put_log()
    logs.append(datetime.utcnow().isoformat())
    upload_put_log(logs)

# Function to count recent PUT requests
def count_recent_put_requests():
    logs = download_put_log()
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    recent_logs = [log for log in logs if datetime.fromisoformat(log) > cutoff_time]
    upload_put_log(recent_logs)  # Save only recent logs back to S3
    return len(recent_logs)

# Function to count inodes
def count_inodes():
    """Count the number of inodes used on the EC2 instance."""
    result = os.popen("df -i / | tail -1 | awk '{print $3}'").read().strip()
    return int(result) if result.isdigit() else 0

# Function to upload a file to S3
def upload_to_s3(file_or_dir, s3_prefix=""):
    """
    Upload a file to S3, with checks for iNodes, PUT requests, and number of files.
    """
    # Check iNodes usage
    inode_count = count_inodes()
    if inode_count > 1000:
        raise Exception(f"iNode limit exceeded on EC2 instance. Current iNodes: {inode_count}")
    
    # Check recent PUT requests
    recent_puts = count_recent_put_requests()
    if recent_puts > 200:
        raise Exception(f"Exceeded 200 PUT requests in the last 24 hours. Current count: {recent_puts}")
    
    # Upload file to S3
    s3_key = os.path.join(s3_prefix, os.path.basename(file_or_dir)).replace("\\", "/")
    s3.upload_file(file_or_dir, s3_bucket_name, s3_key)
    log_put_request()
    print(f"Uploaded {file_or_dir} to s3://{s3_bucket_name}/{s3_key}")

# Paths and parameters for DZI creation
slide_path = (
    "/media/hdd3/neo/error_slides_ndpi/H19-6490;S10;MSKM - 2023-12-11 21.02.14.ndpi"
)
tmp_save_path = "/media/hdd3/neo/S3_tmp_dir/test_slide_3.h5"
heatmap_h5_save_path = (
    "/media/hdd3/neo/S3_tmp_dir/heatmaps/test_slide_3_heatmap.h5"
)

# Generate DZI files
dzsave_h5(
    slide_path,
    tmp_save_path,
    tile_size=512,
    num_cpus=32,
    region_cropping_batch_size=256,
)

create_heatmap_to_h5(slide_path, heatmap_h5_save_path)

print("H5 file and heatmap created successfully.")

# Upload the .h5 file to S3 under the specified subfolder
print("Uploading H5 file to S3...")
start_time = time.time()

h5_s3_prefix = f"{s3_subfolder}"
upload_to_s3(tmp_save_path, h5_s3_prefix)

# Upload the heatmap .h5 file to S3 under the specified subfolder
heatmap_h5_s3_prefix = f"{s3_subfolder}/heatmaps"
upload_to_s3(heatmap_h5_save_path, heatmap_h5_s3_prefix)

end_time = time.time()
print("H5 file and heatmap uploaded successfully.")
print(f"Time taken: {end_time - start_time:.2f} seconds.")
