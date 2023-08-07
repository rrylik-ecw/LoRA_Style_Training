import os
import boto3
import tarfile


def download_and_extract_tar_gz_file(bucket_name, object_key, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Download the tar.gz file from S3 to a local file
    local_tar_gz_file_path = os.path.join(destination_folder, os.path.basename(object_key))
    print(os.path.basename(object_key))
    s3_client.download_file(bucket_name, object_key, local_tar_gz_file_path)

    # Extract the contents of the tar.gz file
    with tarfile.open(local_tar_gz_file_path, 'r:gz') as tar:
        tar.extractall(destination_folder)

    # Remove the downloaded tar.gz file if needed
    os.remove(local_tar_gz_file_path)
    
    
def upload_file_to_s3(local_file_path, bucket_name, subfolder_name, s3_destination_filename=None):
    """
    Uploads a local file to an S3 bucket subfolder.

    Parameters:
    local_file_path (str): The path to the local file.
    bucket_name (str): The name of the S3 bucket.
    subfolder_name (str): The name of the subfolder within the bucket.
    s3_destination_filename (str): (Optional) The filename to be used in S3. If not provided, the original filename will be used.

    Returns:
    bool: True if the upload was successful, False otherwise.
    """
    try:
        # Create an S3 client
        s3_client = boto3.client('s3')

        # Construct the S3 destination path
        if s3_destination_filename is None:
            s3_destination_filename = local_file_path.split('/')[-1]  # Use the original filename if not provided
        s3_destination_path = f"{subfolder_name}/{s3_destination_filename}"

        # Upload the file to S3
        s3_client.upload_file(local_file_path, bucket_name, s3_destination_path)

        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False