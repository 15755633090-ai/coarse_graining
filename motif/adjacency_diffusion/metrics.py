"""All valid non-diagonal pairs, not just true bonds; metrics use upper triangle."""
import torch

BINS = ("all", "t_0_10pct", "t_10_50pct", "t_50_100pct")


@torch.no_grad()
def recovery_statistics(target, restored, noise, epsilon, condition, t, horizon):
    upper = torch.triu(condition.pairs, diagonal=1)
    truth, predicted = target >= .5, restored >= .5
    pair_error, epsilon_error = (restored-target).square(), (epsilon-noise).square()
    fraction = t.float() / horizon
    bins = (torch.ones_like(t, dtype=torch.bool), fraction <= .1,
            (fraction > .1) & (fraction <= .5), fraction > .5)
    tensors, keys = [], []
    for name, graph_mask in zip(BINS, bins):
        mask = upper & graph_mask[:, None, None]
        values = {
            "graphs": graph_mask.sum(), "pairs": mask.sum(), "positive": (mask & truth).sum(),
            "tp": (mask & truth & predicted).sum(), "tn": (mask & ~truth & ~predicted).sum(),
            "fp": (mask & ~truth & predicted).sum(), "fn": (mask & truth & ~predicted).sum(),
            "pair_error_sum": pair_error[mask].sum(), "epsilon_error_sum": epsilon_error[mask].sum(),
            "frobenius_sum": (pair_error * condition.pairs * graph_mask[:, None, None]).sum(),
            "zero_frobenius_sum": (target.square() * condition.pairs * graph_mask[:, None, None]).sum(),
        }
        for key, value in values.items():
            keys.append(name + "/" + key)
            tensors.append(value.double())
    return dict(zip(keys, torch.stack(tensors).cpu().tolist()))


def summarize(statistics):
    result = {}
    for name in BINS:
        def get(key):
            return statistics.get(name + "/" + key, 0)
        graphs, pairs, positive = get("graphs"), get("pairs"), get("positive")
        tp, tn, fp, fn = get("tp"), get("tn"), get("fp"), get("fn")
        recall = tp/(tp+fn) if tp+fn else None
        specificity = tn/(tn+fp) if tn+fp else None
        result[name] = {
            "graphs": int(graphs), "undirected_pairs": int(pairs), "positive_pairs": int(positive),
            "reconstruction_frobenius": get("frobenius_sum")/graphs if graphs else None,
            "zero_adjacency_frobenius": get("zero_frobenius_sum")/graphs if graphs else None,
            "pair_mse": get("pair_error_sum")/pairs if pairs else None,
            "epsilon_mse": get("epsilon_error_sum")/pairs if pairs else None,
            "accuracy": (tp+tn)/pairs if pairs else None,
            "positive_recall": recall, "specificity": specificity,
            "precision": tp/(tp+fp) if tp+fp else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "balanced_accuracy": (recall+specificity)/2 if recall is not None and specificity is not None else None,
        }
    return result
