"""Pre-fusion disagreement on horizontal/vertical neighboring lattice patches."""

import torch


class OverlapDisagreement:
    """O(N) neighbor edges; retain proposals only until their last comparison.

    Each overlapping immediate horizontal/vertical neighbor pair contributes
    equally to the reported means. Diagonal and nonadjacent pairs are excluded.
    This is a local seam diagnostic, not an all-pairs disagreement estimate.
    """

    def __init__(self, layout):
        positions = {(p.y, p.x): p for p in layout.patches}
        ys = sorted({p.y for p in layout.patches})
        xs = sorted({p.x for p in layout.patches})
        self.neighbors = {p.index: set() for p in layout.patches}
        self.patches = {p.index: p for p in layout.patches}
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                a = positions.get((y, x))
                if a is None:
                    continue
                candidates = []
                if iy + 1 < len(ys):
                    candidates.append(positions.get((ys[iy + 1], x)))
                if ix + 1 < len(xs):
                    candidates.append(positions.get((y, xs[ix + 1])))
                for b in candidates:
                    if b is not None and b.y < a.y + a.size and b.x < a.x + a.size:
                        self.neighbors[a.index].add(b.index)
                        self.neighbors[b.index].add(a.index)
        self.pending = {}
        self.seen = set()
        self.maes = []
        self.rmses = []

    def add(self, patch, proposal):
        if patch.index in self.seen:
            raise ValueError("Duplicate proposal in overlap diagnostics")
        self.seen.add(patch.index)
        for other_index in self.neighbors[patch.index]:
            if other_index not in self.pending:
                continue
            other = self.patches[other_index]
            value = self.pending[other_index]
            y0, x0 = max(patch.y, other.y), max(patch.x, other.x)
            y1, x1 = min(patch.y + patch.size, other.y + other.size), min(patch.x + patch.size, other.x + other.size)
            a = proposal[..., y0-patch.y:y1-patch.y, x0-patch.x:x1-patch.x].float()
            b = value[..., y0-other.y:y1-other.y, x0-other.x:x1-other.x].to(a)
            difference = a - b
            self.maes.append(float(difference.abs().mean()))
            self.rmses.append(float(difference.square().mean().sqrt()))
            if self.neighbors[other_index] <= self.seen:
                del self.pending[other_index]
        if not self.neighbors[patch.index] <= self.seen:
            self.pending[patch.index] = proposal.detach().clone()

    def values(self):
        return {
            "overlap_pair_count": len(self.maes),
            "overlap_mae_mean": sum(self.maes) / len(self.maes) if self.maes else 0.0,
            "overlap_mae_max": max(self.maes, default=0.0),
            "overlap_rmse_mean": sum(self.rmses) / len(self.rmses) if self.rmses else 0.0,
        }
