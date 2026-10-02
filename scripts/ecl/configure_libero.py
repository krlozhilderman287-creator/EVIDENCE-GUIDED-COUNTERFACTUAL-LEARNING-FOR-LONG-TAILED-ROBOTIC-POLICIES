"""Write a project-local LIBERO path configuration without interactive prompts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
base = ROOT / "LIBERO/libero/libero"
config = {"benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
          "init_states": str(base / "init_files"), "assets": str(base / "assets"),
          "datasets": str(ROOT / "libero_raw")}
target = ROOT / ".libero/config.yaml"
target.parent.mkdir(exist_ok=True)
# JSON is valid YAML, and this bootstrap needs no third-party modules.
target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(f"LIBERO_CONFIG_PATH={target.parent}")
