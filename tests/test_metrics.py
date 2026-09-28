import numpy as np

from agriopt.metrics import hypervolume_2d, igd_plus, additive_epsilon, nondominated


def test_metrics_identical_front():
    p = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    assert hypervolume_2d(p, (1.1, 1.1)) > 0
    assert igd_plus(p, p) == 0
    assert additive_epsilon(p, p) == 0
    assert len(nondominated(p)) == 3
