"""SEN12MS-CR per-band evaluation entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sen12mscr.evaluate_bands import main

if __name__ == "__main__":
    main()
