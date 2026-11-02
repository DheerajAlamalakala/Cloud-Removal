"""Historical RICE2 U-Net training entry point."""
from workflow_dispatch import dispatch

if __name__ == "__main__":
    dispatch("train_unet.py", default="rice2")
