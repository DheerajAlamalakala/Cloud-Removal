"""Compatibility import for notebooks written against the original RICE2 layout."""
from legacy.rice2.losses import *  # noqa: F401,F403
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sen12mscr.losses import *  # noqa: F401,F403,E402
