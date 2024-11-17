import os
import boto3
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from the .env file
load_dotenv()

# Fetch AWS credentials and bucket name from .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# S3 key for the PUT log file
S3_PUT_LOG_KEY = "put_log/s3_put_log.txt"

# Create S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def download_put_log():
    """Download the PUT log file from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_PUT_LOG_KEY)
        return response["Body"].read().decode("utf-8").splitlines()
    except s3_client.exceptions.NoSuchKey:
        return []

def upload_put_log(logs):
    """Upload the updated PUT log file back to S3."""
    log_data = "\n".join(logs)
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=S3_PUT_LOG_KEY, Body=log_data)

def log_put_request():
    """Log a PUT request with a timestamp."""
    logs = download_put_log()
    logs.append(datetime.utcnow().isoformat())
    upload_put_log(logs)

def count_recent_put_requests():
    """Count PUT requests made in the last 24 hours."""
    logs = download_put_log()
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    recent_logs = [log for log in logs if datetime.fromisoformat(log) > cutoff_time]
    upload_put_log(recent_logs)  # Save only recent logs back to S3
    return len(recent_logs)

def count_inodes():
    """Count the number of inodes used on the EC2 instance."""
    result = os.popen("df -i / | tail -1 | awk '{print $3}'").read().strip()
    return int(result) if result.isdigit() else 0

def upload_to_s3(file_or_dir, s3_prefix=""):
    """
    Upload a file or directory to S3, with checks for iNodes, PUT requests, and number of files.
    
    :param file_or_dir: Path to the file or directory to upload.
    :param s3_prefix: Prefix in the S3 bucket to upload to.
    :return: None or raises an Exception.
    """
    # Check iNodes usage
    inode_count = count_inodes()
    if inode_count > 1000:
        raise Exception(f"iNode limit exceeded on EC2 instance. Current iNodes: {inode_count}")
    
    # Check recent PUT requests
    recent_puts = count_recent_put_requests()
    if recent_puts > 200:
        raise Exception(f"Exceeded 200 PUT requests in the last 24 hours. Current count: {recent_puts}")
    
    # Check number of files
    files_to_upload = []
    if os.path.isdir(file_or_dir):
        for root, _, files in os.walk(file_or_dir):
            for file in files:
                files_to_upload.append(os.path.join(root, file))
    elif os.path.isfile(file_or_dir):
        files_to_upload.append(file_or_dir)
    else:
        raise Exception(f"Path '{file_or_dir}' does not exist or is not a valid file/directory.")
    
    if len(files_to_upload) > 100:
        raise Exception(f"Attempting to upload more than 100 files. Current file count: {len(files_to_upload)}")
    
    # Upload files
    for file_path in files_to_upload:
        s3_key = os.path.join(s3_prefix, os.path.relpath(file_path, file_or_dir)).replace("\\", "/")
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        log_put_request()  # Log the PUT request
    
    print(f"Successfully uploaded {len(files_to_upload)} files to S3 bucket '{S3_BUCKET_NAME}' under prefix '{s3_prefix}'.")
