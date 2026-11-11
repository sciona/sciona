"""Run corrected source packing through all full-size CHAMPS graph variants."""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciona.champs_source_corrections import correct_triplet_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins_bytes = (repo / "docs/reviews/competition_champs_source_pins.json").read_bytes()
    assert hashlib.sha256(pins_bytes).hexdigest() == "99fde10617f64e20fc0459e73723ded08e3692518afd6e74bb67b8f96de804ca"
    pins = json.loads(pins_bytes)
    hashes = {p["software_path"]: p["sha256"] for p in pins["pins"]}
    with tempfile.TemporaryDirectory(prefix="champs-synthetic-") as temporary:
        tmp = Path(temporary)
        subprocess.run([sys.executable, str(repo / "scripts/validate_champs_packing.py"),
                        "--source", str(args.source), "--output", str(tmp / "packing.json"),
                        "--corrected", "--tensors", str(tmp / "tensors.npz")], check=True)
        with np.load(tmp / "tensors.npz", allow_pickle=False) as archive:
            inputs = tuple(torch.from_numpy(archive[f"tensor_{i}"].copy()) for i in range(1,9))
        packing = json.loads((tmp / "packing.json").read_text())
    # Deliberately choose valid categorical IDs larger than the atom count.
    # They must remain embedding IDs and never become geometric references.
    inputs[4][0,:2,1] = 118
    assert inputs[4].shape[-1] == 7
    assert torch.equal(inputs[4][0,:2,5:7], torch.tensor([[0,1],[1,2]]))
    config_bytes = (args.source / "config/models.json").read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == hashes["config/models.json"]
    catalog = json.loads(config_bytes)
    torch.set_num_threads(2)
    results = []
    for name in catalog["names"]:
        folder = "models/" + catalog[name + "_dir"]
        raw = (args.source / folder / "config").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == hashes[folder + "/config"]
        config = ast.literal_eval(raw.decode())
        config.pop("name", None)
        config["dim"] = config.pop("d_model")
        config.update({k:v for k,v in catalog.items() if k.startswith("num_")})
        ns = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
              "np": np, "weight_norm": torch.nn.utils.weight_norm, "colored": lambda text,*_: text}
        for relative in ["modules/hierarchical_embedding.py", "modules/embeddings.py", "graph_transformer.py"]:
            path = folder + "/" + relative
            source = (args.source / path).read_bytes()
            assert hashlib.sha256(source).hexdigest() == hashes[path]
            if relative == "graph_transformer.py":
                source = correct_triplet_layout(source, hashes[path], kind="graph").encode()
            definitions = [n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
            exec(compile(ast.Module(body=definitions,type_ignores=[]), path, "exec"), ns)
        torch.manual_seed(1729)
        np.random.seed(1729)
        model = ns["GraphTransformer"](**config)
        model.eval()
        prediction, _ = model(*inputs)
        assert prediction.shape == (1,68,406) and torch.isfinite(prediction).all()
        # Supervise only the three synthetic requested couplings; padding is
        # excluded. This probes connectivity, not the original training loss.
        loss = prediction[0,0,:3].square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        item = {"variant": folder.split("/")[-1], "parameters": sum(p.numel() for p in model.parameters()),
                "full_padding": True, "high_subtype_id": 118, "finite_forward_backward": True}
        results.append(item)
        print(json.dumps(item), flush=True)
        del model, prediction, loss, grads, ns, _
        gc.collect()
    assert len(results) == 13
    args.output.write_text(json.dumps({"source_commit": pins["commit"], "results": results,
        "packer_checks": packing["checks"],
        "scope": "Corrected prepared-table packing through all13 full configured networks with original padding. No raw chemistry, original training schedule, or pretrained prediction parity claim.",
        "correction_module_sha256": hashlib.sha256((repo / "sciona/champs_source_corrections.py").read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+"\n")


if __name__ == "__main__":
    main()
