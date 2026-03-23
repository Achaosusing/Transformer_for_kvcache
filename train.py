#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trainer import Trainer
from utils import load_yaml, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", default="configs/base_config.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
