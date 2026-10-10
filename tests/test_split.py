import numpy as np

from models import grouped_split


def test_grouped_split_has_no_patient_overlap():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 2000, 10_000)
    y = (rng.random(10_000) < 0.11).astype(int)
    tr, te = grouped_split(y, groups, 5, 42)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert 0.15 < len(te) / len(y) < 0.25
    assert abs(y[te].mean() - y[tr].mean()) < 0.02
