import json
import numpy as np

from zkuc.core.r1cs_io import load_r1cs_json
from zkuc.core.witness_io import assemble_full_z
from zkuc.core.jacobian import matvec_rows_modp, jacobian_submatrix_dense_modp
from zkuc.core.rank_modp import algebraic_rank_modp

def test_add_algebraic_rank(tmp_path):
    # tiny add in snarkjs dict form: 0 = a + b - c
    obj = {
        "prime": str(21888242871839275222246405745257275088548364400416034343698204186575808495617),
        "nVars": 4,
        "nOutputs": 1,
        "nPubInputs": 0,
        "nPrvInputs": 2,
        "nConstraints": 1,
        "constraints": [
            [ {}, {}, {"1":"-1","2":"1","3":"1"} ]
        ],
        "map": [0,1,2,3]
    }
    r1 = tmp_path / "add.r1cs.json"
    r1.write_text(json.dumps(obj))

    R = load_r1cs_json(str(r1))
    p = R.prime

    # witness ["1","8","3","5"] -> c=8, a=3, b=5
    wit = [1, 8, 3, 5]
    z = assemble_full_z(wit, R.var_map, R.n_vars, p)

    # Compute row-wise products Az, Bz over F_p
    Az = matvec_rows_modp(R.A_rows, z, p)
    Bz = matvec_rows_modp(R.B_rows, z, p)

    # Freeze const (0); test rank on columns {1,2,3}
    cols = [1, 2, 3]
    J_sub = jacobian_submatrix_dense_modp(R.A_rows, R.B_rows, R.C_rows, Az, Bz, cols, p)

    # Exact algebraic rank over F_p should be 1 for the adder
    r = algebraic_rank_modp(J_sub, p)
    assert r == 1
