"""Probe original CHAMPS label encoding and packing using synthetic tables."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciona.champs_source_corrections import correct_triplet_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrected", action="store_true")
    parser.add_argument("--tensors", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo / "docs/reviews/competition_champs_source_pins.json").read_text())
    expected = {p["software_path"]: p["sha256"] for p in pins["pins"]}
    raw = (args.source / "src/pipeline_pre.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected["src/pipeline_pre.py"]
    if args.corrected:
        raw = correct_triplet_layout(raw, expected["src/pipeline_pre.py"], kind="packer").encode()
    wanted = {"_create_embedding", "add_embedding", "create_dataset"}
    definitions = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(definitions) == len(wanted)
    ns = {"np": np, "pd": pd, "MAX_ATOM_COUNT": 29, "MAX_BOND_COUNT": 406,
          "MAX_TRIPLET_COUNT": 54, "MAX_QUAD_COUNT": 117}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "<source-packing>", "exec"), ns)
    atoms = pd.DataFrame([dict(molecule_name="synthetic", atom="C", atom_index=i,
                              labeled_atom="C_2_2", x=x, y=y, z=z, angle=.5, charge=0.)
                          for i, (x,y,z) in enumerate([(0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(1.,1.,1.)])])
    pairs = [(0,1),(1,2),(2,3),(0,2),(0,3),(1,3)]
    bonds = pd.DataFrame([dict(molecule_name="synthetic", atom_index_0=a, atom_index_1=b,
                              type="synthetic-type", labeled_type="synthetic-subtype", sublabel_type="synthetic-detail",
                              predict=int(i<3), bond_order=int(i<3), scalar_coupling_constant=float(i),
                              sc_mean=1., sc_std=2., id=i)
                          for i,(a,b) in enumerate(pairs)])
    triplets = pd.DataFrame([dict(molecule_name="synthetic", atom_index_0=a, atom_index_1=b,
                                 atom_index_2=c, label="C2.0-C2.0-C2.0", angle=.5)
                             for a,b,c in [(1,0,2),(2,1,3)]])
    quads = pd.DataFrame([dict(molecule_name="synthetic", atom_index_0=1, atom_index_1=2,
                              atom_index_2=0, atom_index_3=3, label="synthetic-quad", angle=.25)])
    vocabulary = ns["add_embedding"](atoms,bonds,triplets,quads)
    assert all(v["<None>"] == 0 for v in vocabulary.values())
    with contextlib.redirect_stderr(io.StringIO()):
        packed = ns["create_dataset"](atoms,bonds,triplets,quads)
    _, xa, xp, xb, xd, xt, ta, xq, qa, y = packed
    assert xa.shape == (1,29,3) and xb.shape == (1,406,5)
    assert xt.shape == (1,54,7) and xq.shape == (1,117,10)
    assert torch.equal(xd[0,:3], torch.ones(3))
    assert torch.count_nonzero(xa[0,4:]) == 0
    assert torch.count_nonzero(xb[0,6:]) == 0
    assert torch.equal(xt[0,:2,2:5], torch.tensor([[1,0,2],[2,1,3]]))
    # Reproduce the original overwrite: second bond goes into column 5 again.
    assert torch.equal(xt[0,:2,5:7], torch.tensor([[0,1],[1,2]] if args.corrected else [[1,0],[2,0]]))
    assert torch.equal(xq[0,0,5:], torch.tensor([1,0,2,0,1]))
    assert torch.equal(y[0,:3,3], torch.ones(3))
    assert not torch.count_nonzero(y[0,3:])
    # A new vocabulary entry fails rather than silently mapping to padding.
    try:
        ns["add_embedding"](atoms.assign(atom="unseen"), bonds, triplets, quads, vocabulary)
    except KeyError:
        unseen_rejected = True
    else:
        raise AssertionError("Expected unseen vocabulary failure")
    model_findings = []
    for variant in sorted((args.source / "models").glob("*/graph_transformer.py")):
        relative = str(variant.relative_to(args.source))
        source = variant.read_bytes()
        assert hashlib.sha256(source).hexdigest() == expected[relative]
        text = source.decode()
        assert "a1,a2,a3 = x_triplet[i,:,1], x_triplet[i,:,2], x_triplet[i,:,3]" in text
        assert "b1,b2 = x_triplet[i,:,4], x_triplet[i,:,5]" in text
        model_findings.append(variant.parent.name)
    assert len(model_findings) == 13
    if args.tensors:
        if not args.corrected:
            raise ValueError("Only corrected synthetic tensors can be exported")
        np.savez(args.tensors, **{f"tensor_{i}": t.numpy() for i,t in enumerate(packed)})
    args.output.write_text(json.dumps({"source_commit": pins["commit"],
        "corrected_packer": args.corrected,
        "checks": {"padding": True, "squared_distances": True, "quad_references": True,
                   "target_mask": True, "unknown_label_rejected": unseen_rejected},
        "source_defects": ["Packer overwrites triplet column 5 with the second bond index, leaving column 6 zero.",
                           "All 13 graph variants read triplet atom/bond columns one position before the documented seven-column layout."],
        "affected_variants": model_findings,
        "scope": "Synthetic prepared tables through original vocabulary and tensor packing; no raw chemistry-to-model or saved-checkpoint parity claim.",
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+"\n")
    print("Synthetic encoding/packing verified; two triplet layout defects reproduced")


if __name__ == "__main__":
    main()
