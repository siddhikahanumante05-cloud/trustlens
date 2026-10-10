"""Helper utility to download a lightweight DFDC slice on Google Colab or Kaggle.
Zero local disk usage: runs entirely inside cloud environment.
"""
import os
import sys
import argparse
import subprocess

def download_dfdc_slice(kaggle_user=None, kaggle_key=None, kaggle_token=None, target_dir="/content/dfdc", part="00"):
    if kaggle_token:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    if kaggle_user and kaggle_key:
        os.environ["KAGGLE_USERNAME"] = kaggle_user
        os.environ["KAGGLE_KEY"] = kaggle_key

    # Check for credentials (either new KAGGLE_API_TOKEN or legacy USERNAME+KEY)
    has_token = "KAGGLE_API_TOKEN" in os.environ or os.path.exists(os.path.expanduser("~/.kaggle/access_token"))
    has_key = ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ) or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))

    if not has_token and not has_key:
        print("ERROR: Kaggle credentials not found!")
        print("Please set KAGGLE_API_TOKEN or place access_token in ~/.kaggle/access_token")
        sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)
    zip_filename = f"dfdc_train_part_{part}.zip"
    print(f"Downloading DFDC part {part} ({zip_filename}) via Kaggle API...")

    cmd_download = [
        "kaggle", "competitions", "download",
        "-c", "deepfake-detection-challenge",
        "-f", zip_filename,
        "-p", target_dir
    ]
    subprocess.check_call(cmd_download)

    zip_path = os.path.join(target_dir, zip_filename)
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} into {target_dir}...")
        subprocess.check_call(["unzip", "-q", "-o", zip_path, "-d", target_dir])
        os.remove(zip_path)
        print("Extraction complete. Dataset ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DFDC training slice on Colab")
    parser.add_argument("--username", type=str, default=None, help="Kaggle Username")
    parser.add_argument("--key", type=str, default=None, help="Kaggle API Key")
    parser.add_argument("--target_dir", type=str, default="/content/dfdc", help="Target extraction directory")
    parser.add_argument("--part", type=str, default="00", help="DFDC train part number (default: 00)")
    args = parser.parse_args()

    download_dfdc_slice(args.username, args.key, args.target_dir, args.part)
