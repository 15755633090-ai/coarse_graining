"""Detached diagnostics; they never change the optimized objective or .grad buffers."""
import torch


def denoise_statistics(errors, graph_ids, motif_counts):
    errors = errors.detach()
    singleton = motif_counts[graph_ids] == 1
    sums = errors.new_zeros(len(motif_counts))
    counts = errors.new_zeros(len(motif_counts))
    sums.index_add_(0, graph_ids, errors)
    counts.index_add_(0, graph_ids, torch.ones_like(errors))
    present = counts > 0
    return {
        "denoise_single_sum": errors[singleton].sum(),
        "denoise_single_count": int(singleton.sum()),
        "denoise_multi_sum": errors[~singleton].sum(),
        "denoise_multi_count": int((~singleton).sum()),
        "denoise_molecule_sum": (sums[present] / counts[present]).sum(),
        "denoise_molecule_count": int(present.sum()),
    }


def _norm(gradients):
    values = [g.detach().float().square().sum() for g in gradients if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def gradient_diagnostics(supervised, auxiliary, parameters, auxiliary_weight):
    """Use the existing forward graph without consuming RNG or accumulating .grad.

    Shared norms use parameters reached by both losses in this batch. Zero gradients
    remain included; absent paths (e.g. no motif edges) are excluded from the overlap.
    """
    parameters = tuple(p for p in parameters if p.requires_grad)
    prop = torch.autograd.grad(supervised, parameters, retain_graph=True, allow_unused=True)
    aux = (torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
           if auxiliary_weight > 0 else (None,) * len(parameters))
    shared = [i for i, (p, a) in enumerate(zip(prop, aux)) if p is not None and a is not None]
    prop_norm, aux_norm = _norm(prop), _norm(aux)
    prop_shared, aux_shared = _norm([prop[i] for i in shared]), _norm([aux[i] for i in shared])
    weighted = auxiliary_weight * aux_norm
    weighted_shared = auxiliary_weight * aux_shared
    return {
        "property_gradient_norm": prop_norm,
        "auxiliary_gradient_norm_raw": aux_norm,
        "auxiliary_gradient_norm_weighted": weighted,
        "auxiliary_to_property_gradient_ratio": weighted / prop_norm if prop_norm > 0 else None,
        "shared_property_gradient_norm": prop_shared,
        "shared_auxiliary_gradient_norm_weighted": weighted_shared,
        "shared_auxiliary_to_property_gradient_ratio": weighted_shared / prop_shared if prop_shared > 0 else None,
        "shared_gradient_parameters": sum(parameters[i].numel() for i in shared),
    }


def summarize_steps(logs):
    result = {}
    # Explicit denominators handle diagnostic sampling and absent loss groups.
    keys = set().union(*(row.keys() for row in logs))
    for key in sorted(keys):
        values = [row[key] for row in logs if row.get(key) is not None]
        if values:
            result[key] = sum(values) / len(values)
    for group in ("single", "multi", "molecule"):
        count = sum(r.get(f"denoise_{group}_count", 0) for r in logs)
        total = sum(r.get(f"denoise_{group}_sum", 0) for r in logs)
        result[f"denoise_{group}_count"] = count
        result[f"denoise_{group}_sum"] = total
        result[f"denoise_{group}_mean"] = total / count if count else None
    selected = result["denoise_single_count"] + result["denoise_multi_count"]
    result["denoise_selected_mean"] = (
        (result["denoise_single_sum"] + result["denoise_multi_sum"]) / selected if selected else None
    )
    result["gradient_diagnostic_samples"] = sum("property_gradient_norm" in row for row in logs)
    result["optimizer_steps"] = len(logs)
    result["parameters_with_grad_max"] = max(r.get("parameters_with_grad", 0) for r in logs)
    return result
