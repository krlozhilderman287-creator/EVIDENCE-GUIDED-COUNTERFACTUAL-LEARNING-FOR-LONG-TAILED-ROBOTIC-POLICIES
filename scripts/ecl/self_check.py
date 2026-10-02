"""CPU mathematical tests and source compilation; does not load checkpoints."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/ecl"))
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    targets = list((ROOT / "experiments/ecl").rglob("*.py")) + list((ROOT / "scripts/ecl").glob("*.py"))
    for path in targets:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Parsed {len(targets)} ECL Python files", flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
