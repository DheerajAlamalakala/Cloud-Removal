"""Historical RICE2 split utility entry point."""
from workflow_dispatch import dispatch

if __name__ == "__main__":
    dispatch("split_data.py", default="rice2")
