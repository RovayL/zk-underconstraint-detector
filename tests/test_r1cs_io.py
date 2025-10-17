from zkuc.core.r1cs_io import load_r1cs_json
from zkuc.features.structural import structural_features

def test_basic_parse():
    r = load_r1cs_json("circuits/examples/tiny_add.r1cs.json")
    assert r.n_constraints == 2
    assert r.n_vars >= 4

def test_features():
    r = load_r1cs_json("circuits/examples/tiny_add.r1cs.json")
    feats = structural_features(r)
    assert "mult_share" in feats
    assert "avg_fanin" in feats
