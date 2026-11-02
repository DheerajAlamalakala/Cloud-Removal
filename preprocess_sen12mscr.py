"""Compatibility entry point for the earlier SEN12MS-CR preprocessing utility."""
from workflow_dispatch import dispatch

if __name__ == "__main__":
    dispatch("preprocess_sen12mscr.py", default="rice2")
