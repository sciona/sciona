"""Run raw synthetic chemistry through all CHAMPS variants and source loss."""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciona.champs_source_corrections import correct_triplet_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins_bytes = (repo / "docs/reviews/competition_champs_source_pins.json").read_bytes()
    assert hashlib.sha256(pins_bytes).hexdigest() == "99fde10617f64e20fc0459e73723ded08e3692518afd6e74bb67b8f96de804ca"
    pins = json.loads(pins_bytes)
    hashes = {p["software_path"]: p["sha256"] for p in pins["pins"]}
    with tempfile.TemporaryDirectory(prefix="champs-synthetic-") as temporary:
        tmp = Path(temporary)
        subprocess.run([sys.executable, str(repo / "scripts/validate_champs_preprocessing.py"),
                        "--source", str(args.source), "--output", str(tmp / "packing.json"),
                        "--tensors", str(tmp / "tensors.npz")], check=True)
        with np.load(tmp / "tensors.npz", allow_pickle=False) as archive:
            targets = torch.from_numpy(archive["tensor_9"].copy())
            inputs = tuple(torch.from_numpy(archive[f"tensor_{i}"].copy()) for i in range(1,9))
        packing = json.loads((tmp / "packing.json").read_text())
    assert inputs[4].shape[-1] == 7
    raw = (args.source / "src/train.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hashes["src/train.py"]
    loss_def = [n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name == "loss"]
    assert len(loss_def) == 1
    loss_ns = {"torch":torch,"NUM_BOND_ORIG_TYPES":8,"args":SimpleNamespace(champs_loss=True)}
    exec(compile(ast.Module(body=loss_def,type_ignores=[]),"<source-loss>","exec"),loss_ns)
    requested = targets[:,:,3] > 0
    assert int(requested.sum()) == 12
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
        assert prediction.shape == (2,68,406) and torch.isfinite(prediction).all()
        abs_error, type_error, type_count = loss_ns["loss"](prediction,targets,inputs[2])
        assert int(type_count.sum()) == 12
        # Independent scalar indexing reference for subtype gathering/unscaling.
        expected = sum((prediction[m,int(inputs[2][m,b,1])-1,b]*targets[m,b,2]
                        +targets[m,b,1]-targets[m,b,0]).abs()
                       for m,b in requested.nonzero().tolist())
        torch.testing.assert_close(abs_error,expected)
        torch.testing.assert_close(type_error.sum(),expected)
        present = type_count > 0
        log_loss = torch.log(type_error[present]/type_count[present] + 1e-9).mean()
        assert torch.isfinite(log_loss)
        loss = abs_error/type_count.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
        item = {"variant": folder.split("/")[-1], "parameters": sum(p.numel() for p in model.parameters()),
                "full_padding": True, "raw_synthetic_molecules":2, "requested_couplings":12,
                "source_loss_matches_scalar_reference":True, "finite_forward_backward": True}
        if args.checkpoints:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            parameter = next(p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad))
            before = parameter.detach().clone()
            optimizer.step()
            assert not torch.equal(parameter,before)
            optimizer.zero_grad(set_to_none=True)
            del before, optimizer
            with torch.no_grad():
                expected_prediction = model(*inputs)[0]
            with tempfile.TemporaryDirectory(prefix="champs-synthetic-state-") as temporary:
                checkpoint = Path(temporary) / "state.pt"
                torch.save(model.state_dict(),checkpoint)
                with torch.no_grad():
                    parameter.add_(1.)
                restored = torch.load(checkpoint,weights_only=True,mmap=True)
                model.load_state_dict(restored,strict=True)
                del restored
                with torch.no_grad():
                    restored_prediction = model(*inputs)[0]
                torch.testing.assert_close(restored_prediction,expected_prediction,rtol=0,atol=0)
            del parameter, expected_prediction, restored_prediction
            item.update(sgd_parameter_update=True, strict_state_roundtrip=True,
                        restored_predictions_exact=True)
        results.append(item)
        print(json.dumps(item), flush=True)
        del model, prediction, loss, grads, ns, _, abs_error, type_error, expected, log_loss
        gc.collect()
    assert len(results) == 13
    args.output.write_text(json.dumps({"source_commit": pins["commit"], "results": results,
        "raw_preprocessing": packing,
        "checkpoint_mode": "Strict tensor state_dict with weights_only loading after synthetic SGD update" if args.checkpoints else "not_checked",
        "scope": "Raw synthetic chemistry, reserved fresh type vocabulary, corrected packing, all13 full configured networks and original loss versus scalar reference. No original training schedule or pretrained prediction parity claim.",
        "correction_module_sha256": hashlib.sha256((repo / "sciona/champs_source_corrections.py").read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+"\n")


if __name__ == "__main__":
    main()
