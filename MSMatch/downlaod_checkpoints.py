from dotenv import load_dotenv
import os
from zipfile import ZipFile
import gdown

load_dotenv()  # take environment variables from .env.

MSMATCH_DIR = os.getenv("homepath")


def download_and_unzip_checkpoint():
    """
    Downloads a zip file from Google Drive and extracts its contents.

    Returns:
    None
    """
    # Step 1: Download the zip file from Google Drive
    url = "https://drive.google.com/file/d/1TeqmMy0wyN6wpZgc8_hFzxlnlvtvdAp9/view?usp=share_link"
    idFile = url.split("/")[-2]
    downUrl = f"https://drive.google.com/uc?id={idFile}"
    print(downUrl)  # Print the download URL
    output = os.path.join(MSMATCH_DIR, "checkpoints.zip")
    gdown.download(downUrl, output, quiet=False)

    # Step 2: Unzip the downloaded file
    with ZipFile(output, "r") as zipObj:
        # Extract all the contents of the zip file in the specified directory
        destination_dir = MSMATCH_DIR
        zipObj.extractall(destination_dir)

    # Step 3: Delete the zip file
    os.remove(output)
    print("Done!")


if __name__ == "__main__":
    download_and_unzip_checkpoint()
