import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _report_module():
    path = ROOT / "scripts" / "refresh_paper_artifacts.py"
    spec = importlib.util.spec_from_file_location("refresh_paper_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paired_inference_is_reproducible_and_seed_matched():
    module = _report_module()
    rows = []
    for method, shift in [("nsga2", 0.00), ("nsga2_qp", 0.04), ("hnsga2", 0.02)]:
        for seed in range(2026, 2036):
            rows.append({
                "instance": "I01_micro", "method": method, "seed": seed,
                "normalized_hypervolume": 0.7 + shift + 0.001 * (seed - 2026),
                "igd_plus": 0.2 - shift + 0.001 * (seed - 2026),
                "additive_epsilon": 0.3 - shift + 0.001 * (seed - 2026),
                "runtime_s": 50.0 + 100.0 * shift + seed - 2026,
            })
    seed_metrics = pd.DataFrame(rows)
    first = module.paired_inference(seed_metrics, bootstrap_repetitions=500)
    second = module.paired_inference(seed_metrics, bootstrap_repetitions=500)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert set(first.n_pairs) == {10}
    assert first.p_holm.between(0.0, 1.0).all()


def test_primary_reference_is_pooled_while_scip_reference_remains_secondary():
    module = _report_module()
    fronts = pd.DataFrame([
        {"instance": "I01_micro", "method": "exact", "seed": 2026, "income_mcop": 10.0, "risk": 10.0},
        {"instance": "I01_micro", "method": "exact", "seed": 2026, "income_mcop": 0.0, "risk": 0.0},
        {"instance": "I01_micro", "method": "nsga2", "seed": 2026, "income_mcop": 11.0, "risk": 9.0},
    ])
    runs = pd.DataFrame([
        {"instance": "I01_micro", "method": "exact", "seed": 2026, "runtime_s": 1.0,
         "metadata_path": "", "front_path": "", "events_path": ""},
        {"instance": "I01_micro", "method": "nsga2", "seed": 2026, "runtime_s": 2.0,
         "metadata_path": "", "front_path": "", "events_path": ""},
    ])
    primary, _, sensitivity, _ = module.compute_metrics(fronts, runs)
    exact_primary = primary[primary.method.eq("exact")].iloc[0]
    exact_secondary = sensitivity[sensitivity.method.eq("exact")].iloc[0]
    assert exact_primary.reference_source == "pooled empirical nondominated union"
    assert exact_primary.igd_plus > 0.0
    assert exact_secondary.reference_source == "SCIP incumbent front"
    assert exact_secondary.igd_plus == 0.0
