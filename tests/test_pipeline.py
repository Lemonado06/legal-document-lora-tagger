"""End to end on the tiny model: prepare -> train -> evaluate -> merge -> tag."""
import json
from pathlib import Path

import pytest

from app import cli, dataset

ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline(tmp_path):
    data = tmp_path / "clauses.jsonl"
    cli.main(["prepare", "--source", "synthetic", "--per-label", "20", "--out", str(data)])
    run = tmp_path / "run"
    summary = cli.train_run(str(data), str(run), rank=4, epochs=2, tiny=True, log=lambda *_: None)
    assert summary["params"]["trainable_pct"] < 50
    assert (run / "adapter.pt").exists() and (run / "tiny_base.pt").exists()

    m = cli.evaluate_run(str(run), latency_n=5, log=lambda *_: None)
    assert 0 <= m["tagging"]["accuracy"] <= 1
    assert m["efficiency"]["merge_prediction_agreement"] == 1.0
    assert "Efficiency" in (run / "report.md").read_text()

    cli.main(["merge", "--run", str(run)])
    assert (run / "merged" / "config.json").exists()

    out = tmp_path / "tagged.md"
    cli.main(["tag", "--run", str(run), "--input", str(ROOT / "examples" / "sample_contract.txt"),
              "--out", str(out)])
    tagged = json.loads(out.with_suffix(".json").read_text())
    assert len(tagged) == 10
    assert all(t["label"] in dataset.LABELS for t in tagged)


def test_cli_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    for cmd in ("hello", "prepare", "train", "evaluate", "merge", "tag", "sweep"):
        assert cmd in text


@pytest.mark.slow
def test_real_gpt2_loads_and_extends():
    from app import llm
    tok, model, info = llm.load_base("gpt2", ["payment", "termination"])
    assert model.get_input_embeddings().weight.shape[0] == len(tok) == 50257 + 4


def test_hello_runs(capsys):
    cli.main(["hello"])
    assert "is ready" in capsys.readouterr().out
