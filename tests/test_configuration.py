import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ecl.runtime import write_compatible_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_reload_preserves_pretrained_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrained = root / "pretrained"
            (pretrained / "checkpoints").mkdir(parents=True)
            (pretrained / "config.json").write_text(json.dumps({"vla": {"base_vlm": "registered-model-id"}}))
            output = root / "run"
            output.mkdir()
            write_compatible_config(output, pretrained / "checkpoints/step-000001.pt",
                                    SimpleNamespace(base_vlm="unavailable/local/path", global_batch_size=20),
                                    {"method": "ECL", "inference": "factual_only"})
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["vla"]["base_vlm"], "registered-model-id")
            self.assertEqual(config["vla"]["global_batch_size"], 20)
            self.assertEqual(config["experiment"]["inference"], "factual_only")

    def test_cli_overrides_json_defaults(self):
        spec = importlib.util.spec_from_file_location("ecl_train_entry", ROOT / "experiments/ecl/train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch("sys.argv", ["train.py", "--config", str(ROOT / "configs/ecl_libero_core.json"),
                                "--mu", "0.1", "--stop-gradient"]):
            args = module.parse_args()
        self.assertEqual(args.mu, 0.1)
        self.assertTrue(args.stop_gradient)
        self.assertEqual(args.global_batch_size, 20)
        self.assertEqual(args.lambda_value, 0.3)


if __name__ == "__main__":
    unittest.main()
