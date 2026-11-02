"""Keep the historical RICE2 commands while adding an explicit SEN12MS-CR switch."""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def dispatch(rice_script: str, sen_module: str | None = None, default: str = "rice2") -> None:
    args = sys.argv[1:]
    dataset = default
    for index, arg in enumerate(args):
        if arg == "--dataset":
            if index + 1 >= len(args):
                raise SystemExit("--dataset requires rice2 or sen12mscr")
            dataset = args[index + 1].lower()
            del args[index:index + 2]
            break
        if arg.startswith("--dataset="):
            dataset = arg.partition("=")[2].lower()
            del args[index]
            break

    if dataset == "rice2":
        legacy_root = ROOT / "legacy" / "rice2"
        script = legacy_root / rice_script
        sys.path.insert(0, str(legacy_root))
        sys.argv = [str(script), *args]
        runpy.run_path(str(script), run_name="__main__")
        return

    if dataset == "sen12mscr" and sen_module:
        sys.path.insert(0, str(ROOT / "src"))
        sys.argv = [sys.argv[0], *args]
        importlib.import_module(sen_module).main()
        return

    if dataset == "sen12mscr":
        raise SystemExit(f"{rice_script} is a RICE2 workflow; use train.py --dataset sen12mscr for SEN12MS-CR.")
    raise SystemExit(f"Unknown dataset {dataset!r}; choose rice2 or sen12mscr.")
