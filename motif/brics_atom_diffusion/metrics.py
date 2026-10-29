"""Detached recovery diagnostics; no changes to the epsilon training objective."""
import torch

from .data import ATOM_DIMS

TIMES = ("all", "t_0_10pct", "t_10_50pct", "t_50_100pct")
GROUPS = ("all", "boundary", "internal")
BLOCKS = ("all", "element", "charge", "aromatic")


@torch.no_grad()
def recovery_statistics(x0, xt, noise, prediction, steps, boundary, schedule):
    recovered = schedule.reconstruct(xt.detach(), prediction.detach(), steps)
    epsilon_error = (prediction.detach() - noise).square()
    clean_error = (recovered - x0).square()
    epsilon_blocks, clean_blocks, correct_blocks = [], [], []
    offset = 0
    for width in ATOM_DIMS:
        sl = slice(offset, offset + width)
        epsilon_blocks.append(epsilon_error[:, sl].mean(-1))
        clean_blocks.append(clean_error[:, sl].mean(-1))
        correct_blocks.append(recovered[:, sl].argmax(-1) == x0[:, sl].argmax(-1))
        offset += width
    correct = torch.stack(correct_blocks, -1)
    eps = torch.stack([epsilon_error.mean(-1), *epsilon_blocks], -1)
    clean = torch.stack([clean_error.mean(-1), *clean_blocks], -1)
    accuracy = torch.cat((correct.all(-1, keepdim=True), correct), -1).to(eps.dtype)
    fraction = steps.to(eps.dtype) / schedule.steps
    all_atoms = torch.ones_like(boundary)
    time_masks = (all_atoms, fraction <= 0.1, (fraction > 0.1) & (fraction <= 0.5), fraction > 0.5)
    group_masks = (all_atoms, boundary, ~boundary)
    masks = torch.stack([t & g for t in time_masks for g in group_masks]).to(eps.dtype)
    counts = masks.sum(-1, keepdim=True).expand(-1, len(BLOCKS))
    # One device-to-host transfer for all bins/groups/blocks, rather than per metric.
    packed = torch.stack((masks @ eps, masks @ clean, masks @ accuracy, counts), -1).cpu().tolist()
    result = {}
    for ti, time_name in enumerate(TIMES):
        for gi, group_name in enumerate(GROUPS):
            for bi, block_name in enumerate(BLOCKS):
                prefix = f"recovery/{time_name}/{group_name}/{block_name}"
                for suffix, value in zip(("noise_error_sum", "x0_error_sum", "correct", "atoms"), packed[ti * len(GROUPS) + gi][bi]):
                    result[f"{prefix}/{suffix}"] = value
    for group_name in GROUPS:
        prefix = f"recovery/all/{group_name}/all"
        result[f"{group_name}_error_sum"] = result[f"{prefix}/noise_error_sum"]
        result[f"{group_name}_atoms"] = int(result[f"{prefix}/atoms"])
    return result


def summarize_recovery(statistics):
    output = {}
    for time_name in TIMES:
        output[time_name] = {}
        for group_name in GROUPS:
            output[time_name][group_name] = {}
            for block_name in BLOCKS:
                prefix = f"recovery/{time_name}/{group_name}/{block_name}"
                count = int(statistics.get(prefix + "/atoms", 0))
                output[time_name][group_name][block_name] = {
                    "atoms": count,
                    "noise_mse": statistics[prefix + "/noise_error_sum"] / count if count else None,
                    "x0_mse": statistics[prefix + "/x0_error_sum"] / count if count else None,
                    "joint_accuracy" if block_name == "all" else "accuracy":
                        statistics[prefix + "/correct"] / count if count else None,
                }
    return output
