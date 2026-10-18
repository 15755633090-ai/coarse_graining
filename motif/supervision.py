"""Count sampled graphs and actual reconstruction targets, without extra RNG."""
import copy


def empty():
    return dict(sampled_graphs=0, atom_supervised_graphs=0, bond_supervised_graphs=0,
                effective_graphs=0, zero_target_graphs=0, atoms=0, bonds=0,
                masked_atoms=0, masked_bonds=0)


class SupervisionCounter:
    def __init__(self, steps, bond_weight=1.):
        self.steps = steps; self.bond_weight = bond_weight
        self.by_time = {str(t): empty() for t in range(1, steps+1)}

    def add(self, detail):
        d = self.by_time[str(detail['t'])]
        na, nb = detail['masked_atoms'], detail['masked_bonds']
        effective = bool(na or (self.bond_weight > 0 and nb))
        d['sampled_graphs'] += 1
        d['atom_supervised_graphs'] += int(na > 0)
        d['bond_supervised_graphs'] += int(nb > 0)
        d['effective_graphs'] += int(effective)
        d['zero_target_graphs'] += int(not effective)
        for k in ('atoms', 'bonds', 'masked_atoms', 'masked_bonds'): d[k] += detail[k]

    def merge(self, other):
        if self.steps != other.steps or self.bond_weight != other.bond_weight:
            raise ValueError('Incompatible counters')
        for t in self.by_time:
            for k, value in other.by_time[t].items(): self.by_time[t][k] += value

    def report(self):
        bins = {k: empty() for k in ('low', 'mid', 'high')}
        all_times = empty()
        for ts, d in self.by_time.items():
            # Integer thirds, explicitly defined even for T<3.
            name = ('low', 'mid', 'high')[min(2, 3*(int(ts)-1)//self.steps)]
            for k, v in d.items(): bins[name][k] += v; all_times[k] += v
        return {'all': all_times, 'bins': bins, 't1': copy.deepcopy(self.by_time['1']),
                'by_time': copy.deepcopy(self.by_time),
                'effective_definition': 'masked atoms > 0 OR (bond_weight > 0 AND masked bonds > 0)',
                'bins_definition': 'index=min(2, floor(3*(t-1)/T)); low/mid/high'}
