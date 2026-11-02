"""Compatibility import for the earlier flat SEN12MS-CR laptop layout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sen12mscr.dataset import *  # noqa: F401,F403,E402
