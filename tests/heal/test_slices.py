from shipwright.heal import slices


def test_slice_by_write_count():
    assert slices.slice_for_write_count(0) == slices.FEW_WRITES
    assert slices.slice_for_write_count(1) == slices.FEW_WRITES
    assert slices.slice_for_write_count(2) == slices.MANY_WRITES


def test_slice_pass_metrics_average_per_slice_and_skip_unknown_tasks():
    per_task = {
        "1": {"n_success": 1, "n_decided": 3, "pass_k": 0.0},
        "2": {"n_success": 3, "n_decided": 3, "pass_k": 1.0},
        "3": {"n_success": 0, "n_decided": 3, "pass_k": 0.0},
        "9": {"n_success": 3, "n_decided": 3, "pass_k": 1.0},  # not in the map
    }
    slice_by_task = {"1": slices.FEW_WRITES, "2": slices.FEW_WRITES, "3": slices.MANY_WRITES}
    result = slices.slice_pass_metrics(per_task, slice_by_task)
    assert result[slices.FEW_WRITES]["num_tasks"] == 2
    assert abs(result[slices.FEW_WRITES]["pass_1"] - (1 / 3 + 1) / 2) < 1e-9
    assert result[slices.FEW_WRITES]["pass_k"] == 0.5
    assert result[slices.MANY_WRITES] == {"num_tasks": 1, "pass_1": 0.0, "pass_k": 0.0}


def test_the_real_heldout_split_has_both_slices():
    slice_by_task = slices.write_slice_by_task("retail", "test")
    assert set(slice_by_task.values()) == set(slices.SLICE_NAMES)
