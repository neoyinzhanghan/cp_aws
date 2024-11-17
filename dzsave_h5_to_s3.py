import os
import time
import boto3
from dotenv import load_dotenv
from LLRunner.slide_processing.dzsave_h5 import dzsave_h5
from compute_heatmap import create_heatmap_to_h5
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

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
    logs.append(datetime.now(timezone.utc).isoformat())  # Use timezone-aware datetime
    upload_put_log(logs)

# Function to count recent PUT requests
def count_recent_put_requests():
    logs = download_put_log()
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
    
    valid_logs = []
    for log in logs:
        try:
            # Parse logs as datetime and make timezone-aware if needed
            parsed_log = datetime.fromisoformat(log)
            if parsed_log.tzinfo is None:  # If naive, assume UTC
                parsed_log = parsed_log.replace(tzinfo=timezone.utc)
            valid_logs.append(parsed_log)
        except ValueError:
            print(f"Skipping invalid log entry: {log}")
    
    # Filter logs within the last 24 hours
    recent_logs = [log for log in valid_logs if log > cutoff_time]
    upload_put_log([log.isoformat() for log in recent_logs])  # Save only valid logs
    print(f"Number of Recent PUT requests in the last 24 hours: {len(recent_logs)}")
    return len(recent_logs)

# Function to count objects (akin to "inodes") in S3 under a specific prefix
def count_objects_in_s3():
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=s3_bucket_name, Prefix="")
    object_count = sum(1 for page in pages for _ in page.get("Contents", []))
    print(f"S3 Bucket Name: {s3_bucket_name}")
    print(f"Number of objects in S3: {object_count}")
    return object_count

# Function to upload a file to S3
def upload_to_s3(file_or_dir, s3_prefix=""):
    """
    Upload a file to S3, with checks for S3 object count, PUT requests, and number of files.
    """
    # Check S3 object count under the prefix
    object_count = count_objects_in_s3()
    if object_count > 1000:  # Example limit
        raise Exception(f"S3 object limit exceeded under prefix '{s3_prefix}'. Current objects: {object_count}")
    
    # Check recent PUT requests
    recent_puts = count_recent_put_requests()
    if recent_puts > 200:
        raise Exception(f"Exceeded 200 PUT requests in the last 24 hours. Current count: {recent_puts}")
    
    # Check if the file exists and how many inodes are being uploaded
    if not os.path.exists(file_or_dir):
        raise Exception(f"Path '{file_or_dir}' does not exist.")
    
    if os.path.isdir(file_or_dir):
        files_to_upload = [os.path.join(file_or_dir, f) for f in os.listdir(file_or_dir)]

        print(f"Uploading {len(files_to_upload)} files (inodes/put requests) to S3...")

        if len(files_to_upload) > 100:
            raise Exception(f"Attempting to upload more than 100 files. Current file count: {len(files_to_upload)}")
    
    elif os.path.isfile(file_or_dir):
        files_to_upload = [file_or_dir]

        print(f"Uploading {len(files_to_upload)} file (inode/put request) to S3...")

        if len(files_to_upload) > 100:
            raise Exception(f"Attempting to upload more than 100 files. Current file count: {len(files_to_upload)}")
    
    # Upload file to S3
    s3_key = os.path.join(s3_prefix, os.path.basename(file_or_dir)).replace("\\", "/")
    s3.upload_file(file_or_dir, s3_bucket_name, s3_key)
    log_put_request()
    print(f"Uploaded {file_or_dir} to s3://{s3_bucket_name}/{s3_key}")

# Paths and parameters for DZI creation
slide_path = (
    "/media/hdd3/neo/error_slides_ndpi/H19-6490;S10;MSKM - 2023-12-11 21.02.14.ndpi"
)
tmp_save_path = "/media/hdd3/neo/S3_tmp_dir/test_slide_2.h5"
heatmap_h5_save_path = (
    "/media/hdd3/neo/S3_tmp_dir/heatmaps/test_slide_2_heatmap.h5"
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
