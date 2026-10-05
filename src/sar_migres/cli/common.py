from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.config import load_config, load_dotenv_file, parse_overrides

ROOT = Path(__file__).resolve().parents[3]

def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", nargs="+", default=[str(ROOT / "configs" / "local.yaml")])
    p.add_argument("--set", nargs="*", default=[], metavar="section.key=value")
    p.add_argument("--env-file", default=str(ROOT / ".env"))
    return p

def load(args):
    names = load_dotenv_file(args.env_file)
    if names:
        print(f"loaded {len(names)} variable(s) from {args.env_file}: {', '.join(names)}")
    return load_config(*args.config, overrides=parse_overrides(args.set))
