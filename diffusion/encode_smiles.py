"""Extract frozen node representations and sum/mean pooled graph features."""
import argparse
from pathlib import Path

import torch

from bond_diffusion.data import collate_graphs, graph_from_smiles
from bond_diffusion.trainer import load_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "outputs/ogb_clean/encoder.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/embeddings.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_encoder(args.checkpoint, args.device).requires_grad_(False)
    batch = collate_graphs([graph_from_smiles(s) for s in args.smiles]).to(torch.device(args.device))
    with torch.no_grad():
        nodes = model.encode_nodes(batch.node_features, batch.bonds, batch.node_mask)
        sums = (nodes * batch.node_mask.unsqueeze(-1)).sum(dim=1)
        means = sums / batch.node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = torch.cat([sums, means], dim=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"smiles": args.smiles, "node_embeddings": nodes.cpu(), "node_mask": batch.node_mask.cpu(), "graph_embeddings": pooled.cpu()}, args.output)
    print(f"node_embeddings={tuple(nodes.shape)}, graph_embeddings={tuple(pooled.shape)}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
