"""Shared CLI plumbing for the scripts (adds src/ to sys.path so no install is needed)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_migres.config import load_config, load_dotenv_file, parse_overrides  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", nargs="+", default=[str(ROOT / "configs" / "local.yaml")],
                   help="one or more YAML files, merged in order (default: configs/local.yaml = offline smoke test)")
    p.add_argument("--set", nargs="*", default=[], metavar="section.key=value", help="config overrides")
    p.add_argument("--env-file", default=str(ROOT / ".env"), help="optional .env with API keys (never committed)")
    return p


def load(args):
    names = load_dotenv_file(args.env_file)
    if names:
        print(f"loaded {len(names)} variable name(s) from {args.env_file}: {', '.join(names)}")
    return load_config(*args.config, overrides=parse_overrides(args.set))
