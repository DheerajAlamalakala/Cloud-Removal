"""Backward-compatible evaluation entry point; select SEN12MS-CR with --dataset."""
from workflow_dispatch import dispatch

if __name__ == "__main__":
    dispatch("evaluate.py", "sen12mscr.evaluate", default="rice2")
