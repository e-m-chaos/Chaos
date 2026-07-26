from imu_features import REGISTRY


def test_registry_has_all_families():
    expected = {
        "statistical",
        "magnitude",
        "frequency",
        "geometrical",
        "mechanical",
        "crossaxis",
        "nonlinear",
        "topological",
        "wavelet",
    }
    assert expected.issubset(set(REGISTRY.families()))


def test_registry_keys_are_unique():
    keys = [s.key for s in REGISTRY.all()]
    assert len(keys) == len(set(keys))


def test_registry_has_many_features():
    # a "giant feature engine" should have real breadth
    assert len(REGISTRY) >= 60


def test_by_family_returns_only_that_family():
    specs = REGISTRY.by_family("frequency")
    assert specs
    assert all(s.family == "frequency" for s in specs)
