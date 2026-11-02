"""Historical RICE2 inference entry point. SEN12MS-CR uses inference/app.py."""
from workflow_dispatch import dispatch

if __name__ == "__main__":
    dispatch("inference.py", default="rice2")
