"""Backward-compatible training entry point; select SEN12MS-CR with --dataset."""
from workflow_dispatch import dispatch

if __name__ == "__main__":
    dispatch("train.py", "sen12mscr.train", default="rice2")
