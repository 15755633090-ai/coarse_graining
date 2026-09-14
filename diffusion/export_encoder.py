from __future__ import annotations

import argparse

from bond_diffusion.trainer import export_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the pretrained bonding-aware encoder")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    export_encoder(args.checkpoint, args.output)
    print(f"encoder exported to {args.output}")


if __name__ == "__main__":
    main()
