from pathlib import Path

from agriopt.dane_sipsa import load_product_aliases, canonicalize_product

ROOT = Path(__file__).resolve().parents[1]


def test_product_aliases():
    aliases = load_product_aliases(ROOT / "configs/products.yaml")
    assert canonicalize_product("Piña perolera", aliases) == "Pina"
    assert canonicalize_product("Cebolla junca", aliases) == "Cebolla larga"
