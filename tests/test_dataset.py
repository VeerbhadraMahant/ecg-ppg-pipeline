import numpy as np

from src.data.dataset import record_wise_folds


def test_record_wise_folds_never_splits_a_record():
    rng = np.random.RandomState(0)
    records = np.array([f"rec{i}" for i in range(30)])
    # each record contributes a variable number of windows
    record_id = np.concatenate([np.full(rng.randint(1, 5), r) for r in records])

    for train_idx, val_idx in record_wise_folds(record_id, n_folds=5, seed=0):
        train_records = set(record_id[train_idx])
        val_records = set(record_id[val_idx])
        assert train_records.isdisjoint(val_records)


def test_record_wise_folds_cover_all_windows_exactly_once():
    record_id = np.array([f"rec{i}" for i in range(20) for _ in range(3)])
    n_folds = 4
    seen = np.zeros(len(record_id), dtype=bool)
    for _, val_idx in record_wise_folds(record_id, n_folds=n_folds, seed=1):
        assert not seen[val_idx].any()
        seen[val_idx] = True
    assert seen.all()
