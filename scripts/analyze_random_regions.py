"""Label-free train-split geometry audit for the random region branch."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_encoder.data import OGBMoleculePropertyDataset
from multiscale_tokenizer.random_regions import graph_distances, sample_regions
from multiscale_tokenizer.partition import derive_partition_seed


def describe(values):
    return dict(zip(("min", "p25", "median", "p75", "p95", "max"),
                    np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist()),
                mean=float(np.mean(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/random_regions_lipo_geometry.json")
    args = parser.parse_args()
    dataset = OGBMoleculePropertyDataset("datasets/lipo", split="train")
    result = {"split": "train", "molecules": len(dataset), "views": 5,
              "atoms_per_center": 8, "max_centers": 8, "radii": {}}
    stats = {r: {"N": [], "K": [], "coverage": [], "jaccard": [], "duplicate": [], "union": []} for r in (2, 4)}
    for record in dataset:
        distances = graph_distances(record.graph.bonds)
        for radius, values in stats.items():
            values["N"].append(len(distances))
            for view in range(5):
                seed = derive_partition_seed(derive_partition_seed(100000, record.sample_id, 0), view)
                centers, members = sample_regions(distances, seed, radius)
                values["K"].append(len(centers))
                values["coverage"].extend(members.mean(1).tolist())
                values["union"].append(members.any(0).mean())
                for i in range(len(centers)):
                    for j in range(i):
                        values["jaccard"].append((members[i] & members[j]).sum() / (members[i] | members[j]).sum())
                        values["duplicate"].append(np.array_equal(members[i], members[j]))
    for radius, values in stats.items():
        result["radii"][radius] = {
            "atom_count": describe(values["N"]), "center_count": describe(values["K"]),
            "region_fraction": describe(values["coverage"]),
            "pair_jaccard": describe(values["jaccard"]),
            "duplicate_pair_fraction": float(np.mean(values["duplicate"])),
            "sampled_union_fraction": describe(values["union"]),
            "region_fraction_at_least_70pct": float(np.mean(np.array(values["coverage"]) >= .7)),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
