"""Exercise raw synthetic geometry through the CHAMPS source preprocessing."""
import argparse
import ast
import collections
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciona.champs_source_corrections import correct_triplet_layout
from sciona.champs_ensemble import COUPLING_TYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensors", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo / "docs/reviews/competition_champs_source_pins.json").read_text())
    hashes = {p["software_path"]:p["sha256"] for p in pins["pins"]}
    raw = (args.source / "src/pipeline_pre.py").read_bytes()
    code = correct_triplet_layout(raw, hashes["src/pipeline_pre.py"], kind="packer")
    substitutions = {
        "    import pybel": "    from openbabel import pybel",
        "mol.atoms[i].heavyvalence": "mol.atoms[i].OBAtom.GetHvyDegree()",
        "mol.atoms[i].heterovalence": "mol.atoms[i].OBAtom.GetHeteroDegree()",
        "mol.atoms[i].valence": "mol.atoms[i].OBAtom.GetExplicitDegree()",
        "bond_dataframe.append(new_data,verify_integrity=True,sort=False)":
        "pd.concat([bond_dataframe,new_data],verify_integrity=True,sort=False)",
        'bonds_train.groupby("labeled_type").mean()["scalar_coupling_constant"]':
        'bonds_train.groupby("labeled_type")["scalar_coupling_constant"].mean()',
        'bonds_train.groupby("labeled_type").std()["scalar_coupling_constant"]':
        'bonds_train.groupby("labeled_type")["scalar_coupling_constant"].std()',
    }
    for old,new in substitutions.items():
        assert code.count(old) == 1
        code = code.replace(old,new)
    code = code.replace('.count().max()[0]', '.count().max().iloc[0]')
    tree = ast.parse(code)
    names = {"make_structure_dict", "enhance_structure_dict", "enhance_atoms", "enhance_bonds",
             "add_all_pairs", "make_triplets", "make_quadruplets", "_create_embedding",
             "add_embedding", "get_scaling", "add_scaling", "create_dataset"}
    definitions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(definitions) == len(names)
    source = (args.source / "src/xyz2mol.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == hashes["src/xyz2mol.py"]
    spec = importlib.util.spec_from_file_location("champs_raw_xyz", args.source / "src/xyz2mol.py")
    x2m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x2m)
    ns = {"np":np, "pd":pd, "collections":collections, "itertools":itertools, "x2m":x2m,
          "manual_bond_order_dict": {}, "atomic_num_dict":{"H":1,"C":6,"N":7,"O":8,"F":9},
          "bond_order_dict":{Chem.BondType.SINGLE:1,Chem.BondType.AROMATIC:1.5,Chem.BondType.DOUBLE:2,Chem.BondType.TRIPLE:3},
          "MAX_ATOM_COUNT":29,"MAX_BOND_COUNT":406,"MAX_TRIPLET_COUNT":54,"MAX_QUAD_COUNT":117}
    # Algorithmic label rules from pinned source, not per-record corrections.
    constants = [n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in {"classification_corrections","small_longtypes"} for t in n.targets)]
    exec(compile(ast.Module(body=constants+definitions,type_ignores=[]),"<source-preprocessing>","exec"),ns)
    # Idealized synthetic ethane: two carbons and six hydrogens. No real records.
    coordinates = [[-.77,0.,0.],[.77,0.,0.]]
    for x, phase in [(-1.13,0.),(1.13,np.pi/3)]:
        for j in range(3):
            angle = phase + 2*np.pi*j/3
            coordinates.append([x,1.03*np.cos(angle),1.03*np.sin(angle)])
    symbols = ["C","C"] + ["H"]*6
    atom_rows, bond_rows = [], []
    for case,shift in enumerate([0.,2.]):
        key = f"synthetic-{case}"
        for i,(symbol,xyz) in enumerate(zip(symbols,coordinates)):
            atom_rows.append(dict(molecule_name=key,atom_index=i,atom=symbol,x=xyz[0]+shift,y=xyz[1],z=xyz[2]))
        for index,(carbon,hydrogen) in enumerate([(0,2),(0,3),(0,4),(1,5),(1,6),(1,7)]):
            bond_rows.append(dict(id=case*6+index,molecule_name=key,atom_index_0=hydrogen,
                                  atom_index_1=carbon,type="1JHC",scalar_coupling_constant=float(index+case)))
    atoms,bonds = pd.DataFrame(atom_rows),pd.DataFrame(bond_rows)
    with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
        structures = ns["enhance_structure_dict"](ns["make_structure_dict"](atoms))
        assert all(np.count_nonzero(m["bond_orders"])==14 for m in structures.values())
        atoms = ns["enhance_atoms"](atoms,structures)
        bonds = ns["add_all_pairs"](ns["enhance_bonds"](bonds,structures),structures).reset_index()
        triplets = ns["make_triplets"](list(structures),structures)
        quads = ns["make_quadruplets"](list(structures),structures)
        for frame,keys in [(atoms,["atom_index"]),(bonds,["atom_index_0","atom_index_1"]),
                           (triplets,["atom_index_0","atom_index_1","atom_index_2"]),
                           (quads,["atom_index_0","atom_index_1","atom_index_2","atom_index_3"])]:
            frame.sort_values(["molecule_name"]+keys,inplace=True)
        vocabulary = ns["add_embedding"](atoms,bonds,triplets,quads)
        # The source loss reserves primary IDs 1..8 for supervised types.
        # A tiny synthetic population lacks seven types; reserve their slots
        # explicitly so added non-coupling pairs cannot enter that objective.
        vocabulary[("bond",0)] = ns["_create_embedding"](
            pd.Series(list(bonds["type"].unique()) + list(COUPLING_TYPES)))
        ns["add_embedding"](atoms,bonds,triplets,quads,vocabulary)
        means,stds = ns["get_scaling"](bonds)
        bonds = ns["add_scaling"](bonds,means,stds)
        packed = ns["create_dataset"](atoms,bonds,triplets,quads)
    assert all(torch.isfinite(t).all() for t in packed)
    assert int(packed[-1][:,:,3].sum()) == 12
    supervised = (packed[3][:,:,0] > 0) & (packed[3][:,:,0] <= 8)
    assert torch.equal(supervised, packed[-1][:,:,3] > 0)
    assert triplets.shape[0]==24 and quads.shape[0]==18
    assert torch.equal(packed[5][0],packed[5][1])
    np.testing.assert_allclose(packed[4][0],packed[4][1],atol=1e-6)
    if args.tensors:
        np.savez(args.tensors,**{f"tensor_{i}":t.numpy() for i,t in enumerate(packed)})
    args.output.write_text(json.dumps({"source_commit":pins["commit"],"synthetic_molecules":2,
        "chemical_bonds":14,"triplets":len(triplets),"quadruplets":len(quads),"requested_predictions":12,
        "finite_packed_tensors":True,"translation_checks":True,
        "compatibility_changes":substitutions,"triplet_layout_corrected":True,
        "synthetic_vocabulary_reserves_eight_coupling_types":True,
        "scope":"Raw synthetic chemistry through motifs, fresh vocabulary, scaling and packing. No trained-vocabulary/pretrained-weight parity claim.",
        "validator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+"\n")
    print("Raw synthetic chemistry through tensor packing passed")


if __name__ == "__main__":
    main()
