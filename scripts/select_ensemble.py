"""Pick an ensemble on VALIDATION data only, then score the single winner on the test set.

    python scripts/select_ensemble.py --task hot --device cuda

The candidate list is fixed in advance (not searched over every subset) so that choosing
among them does not overfit the validation split. Only the chosen ensemble is evaluated on
test, once; its folder is artifacts/<task>/best.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hinglishguard.models import calibrate_ensemble, evaluate  # noqa: E402

SEEDS = ["hingroberta", "hingroberta-s13", "hingroberta-s7"]
CANDIDATES = {
    "hingroberta alone": ["hingroberta"],
    "hingaux alone (extra-data pretraining)": ["hingaux"],
    "tfidf+hingroberta (current)": ["tfidf", "hingroberta"],
    "tfidf+hingaux": ["tfidf", "hingaux"],
    "hingaux + hingroberta x3 seeds + tfidf": ["hingaux"] + SEEDS + ["tfidf"],
    "everything": ["hingaux"] + SEEDS + ["hingmixed", "muril", "tfidf"],
    "hingroberta x3 seeds": SEEDS,
    "hingroberta x3 seeds + tfidf": SEEDS + ["tfidf"],
    "3 Hinglish/Indic encoders": ["hingroberta", "hingmixed", "muril"],
    "3 Hinglish/Indic encoders + tfidf": ["hingroberta", "hingmixed", "muril", "tfidf"],
    "all transformers + tfidf": SEEDS + ["hingmixed", "muril", "tfidf"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["hot", "thar"], required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    P, A = ROOT / "data/processed", ROOT / "artifacts"
    results = {}
    for name, members in CANDIDATES.items():
        missing = [m for m in members if not (A / args.task / m / "metadata.json").exists()]
        if missing:
            print(f"skip {name}: missing {missing}")
            continue
        report = calibrate_ensemble(P, A, args.task, args.device, members=tuple(members), output_name="best")
        results[name] = (report["validation"]["macro_f1"], members)
        print(f"[{args.task}] VALIDATION {report['validation']['macro_f1']:.4f}  {name}", flush=True)
    winner = max(results, key=lambda k: results[k][0])
    members = results[winner][1]
    calibrate_ensemble(P, A, args.task, args.device, members=tuple(members), output_name="best")
    test = evaluate(P, A, args.task, "best", args.device)
    hate = {k: round(v["f1-score"], 3) for k, v in test["report"].items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")}
    meta = json.loads((A / args.task / "best/metadata.json").read_text(encoding="utf-8"))
    summary = {"task": args.task, "chosen": winner, "members": members, "validation": {k: round(v[0], 4) for k, v in results.items()},
               "test_macro_f1": round(test["macro_f1"], 4), "test_ci95": [round(x, 4) for x in test["macro_f1_ci95"]],
               "test_per_class_f1": hate, "validation_coverage": round(meta["thresholds"]["coverage"], 3)}
    (A / args.task / "best/selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
