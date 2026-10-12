"""Regression checks for final-size explanatory artwork, without experiments."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_methodology_figures as diagrams


@pytest.fixture
def context():
    diagrams._configure_style()
    config = {"experiment": {"nsga2": {"population": 80, "generations": 150},
              "hybrid": {"refinement_fraction": .25, "warmup_generations": 5,
                         "min_training_samples": 30}, "exact": {"points": 11},
              "seeds": list(range(2026, 2036)),
              "instances": [f"I{i:02d}_example" for i in range(1, 9)]}}
    snapshot = {"included_complete_instances": config["experiment"]["instances"][:4]}
    return snapshot, config


@pytest.mark.parametrize("kind", ["chromosome", "variants", "workflow"])
def test_final_size_layout_has_no_collisions(context, kind):
    snapshot, config = context
    builders = {"chromosome": diagrams.figure_chromosome,
                "variants": lambda: diagrams.figure_variants(config),
                "workflow": lambda: diagrams.figure_workflow(snapshot, config)}
    fig = builders[kind]()
    try:
        diagrams._validate_layout(fig)
        assert fig.get_size_inches()[0] * 25.4 == pytest.approx(diagrams.WIDTH_MM)
        assert min(t.get_fontsize() for ax in fig.axes for t in ax.texts) >= 9
    finally:
        plt.close(fig)


def test_overlapping_boxes_are_rejected():
    fig, ax = diagrams._new_canvas(50)
    diagrams._box(ax, 2, 2, 40, 20, "First")
    diagrams._box(ax, 20, 10, 40, 20, "Second")
    try:
        with pytest.raises(ValueError, match="Overlapping diagram boxes"):
            diagrams._validate_layout(fig)
    finally:
        plt.close(fig)


def test_overflow_is_rejected():
    fig, ax = diagrams._new_canvas(50)
    diagrams._box(ax, 2, 2, 10, 10, "This label cannot fit")
    try:
        with pytest.raises(ValueError, match="Text overflows box"):
            diagrams._validate_layout(fig)
    finally:
        plt.close(fig)


def test_export_is_vector_first_and_final_size(tmp_path, context):
    _, config = context
    fig = diagrams.figure_variants(config)
    outputs = diagrams._save(fig, "variants", tmp_path / "paper", tmp_path / "artifacts")
    assert len(outputs) == 5
    for path in outputs:
        assert path.stat().st_size > 100
    for suffix, dpi in [(".png", 300), ("_600dpi.png", 600)]:
        with Image.open(tmp_path / "artifacts" / f"variants{suffix}") as img:
            assert abs(img.width - diagrams.WIDTH_MM / 25.4 * dpi) <= 1
            assert img.height < 154 / 25.4 * dpi  # outer white margin was cropped
    assert "<text" in (tmp_path / "artifacts" / "variants.svg").read_text(encoding="utf-8")
