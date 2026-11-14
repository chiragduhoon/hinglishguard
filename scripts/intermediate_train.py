"""Stage 1 of intermediate-task training: teach an encoder "abusive/aggressive vs not" on extra
Hindi/Hinglish data, then save the ENCODER ONLY so `cli train --base-model <dir>` can fine-tune it
on HOT or THAR exactly like any Hugging Face model.

    python scripts/intermediate_train.py --device cuda
    python -m hinglishguard.cli train --task hot --backend transformer --device cuda \
        --base-model artifacts/_intermediate/hing-aux --output-name hingaux

Extra data (downloaded to data/raw/aux/, never committed):
  - TRAC-1 Hindi (Kumar et al., 2018; Apache-2.0): Facebook comments, ~74% Roman-script Hinglish.
    OAG (overtly aggressive) -> 1, NAG -> 0. CAG (covert) is dropped as too ambiguous.
  - MACD Hindi (Gupta et al., NeurIPS 2022; no licence stated, so research use only, not
    redistributed): ShareChat comments in Devanagari. Source label 0 = abusive -> 1.

Leakage guard: any extra comment whose normalised fingerprint matches a HOT/THAR validation or
test record is removed before training, so those splits stay unseen.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hinglishguard.data import fingerprint  # noqa: E402
from hinglishguard.normalization import clean  # noqa: E402

AUX = ROOT / "data/raw/aux"
OUT = ROOT / "artifacts/_intermediate/hing-aux"
SOURCES = {
    "trac_hi_train.csv": "https://raw.githubusercontent.com/kmi-linguistics/trac-1/master/hindi/agr_hi_train.csv",
    "trac_hi_dev.csv": "https://raw.githubusercontent.com/kmi-linguistics/trac-1/master/hindi/agr_hi_dev.csv",
    "macd_hindi_train.csv": "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_train.csv",
    "macd_hindi_val.csv": "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_val.csv",
    "macd_hindi_test.csv": "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/hindi_test.csv",
}


def download():
    AUX.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (AUX / name).exists():
            urllib.request.urlretrieve(url, AUX / name)


def load_rows():
    import pandas as pd
    rows = []
    for name in ("trac_hi_train.csv", "trac_hi_dev.csv"):
        d = pd.read_csv(AUX / name, header=None, names=["id", "text", "label"], on_bad_lines="skip")
        d = d[d.label.isin(["OAG", "NAG"])]
        rows += [(str(t), int(lbl == "OAG"), "trac") for t, lbl in zip(d.text, d.label)]
    for name in ("macd_hindi_train.csv", "macd_hindi_val.csv", "macd_hindi_test.csv"):
        d = pd.read_csv(AUX / name)
        rows += [(str(t), int(lbl == 0), "macd") for t, lbl in zip(d.text, d.label)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="l3cube-pune/hing-roberta")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    download()
    records = json.loads((ROOT / "data/processed/records.json").read_text(encoding="utf-8"))
    protected = {fingerprint(r["text"]) for r in records if r["split"] in ("validation", "test")}
    rows, dropped = [], 0
    for text, label, src in load_rows():
        text = clean(text)
        if not text.strip():
            continue
        if fingerprint(text) in protected:
            dropped += 1
            continue
        rows.append({"text": text, "label": label, "source": src})
    by_src = {s: sum(r["source"] == s for r in rows) for s in ("trac", "macd")}
    print(f"aux rows: {len(rows)} {by_src}, positive={sum(r['label'] for r in rows)}, removed as HOT/THAR val/test duplicates: {dropped}", flush=True)

    import random

    import torch
    from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(args.seed)
    random.Random(args.seed).shuffle(rows)
    hold = rows[:2000]  # sanity check only; HOT/THAR splits are what matter
    train = rows[2000:]
    tok = AutoTokenizer.from_pretrained(args.base_model)

    class DS(torch.utils.data.Dataset):
        def __init__(self, r):
            self.r = r

        def __len__(self):
            return len(self.r)

        def __getitem__(self, i):
            return dict(tok(self.r[i]["text"], truncation=True, max_length=192), labels=self.r[i]["label"])

    def compute(p):
        from sklearn.metrics import f1_score
        return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(axis=1), average="macro")}

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    targs = TrainingArguments(output_dir=str(OUT.parent / "checkpoints"), learning_rate=2e-5, per_device_train_batch_size=16,
                              per_device_eval_batch_size=32, num_train_epochs=args.epochs, weight_decay=0.01, warmup_ratio=0.06,
                              eval_strategy="epoch", save_strategy="no", fp16=args.device == "cuda", seed=args.seed,
                              report_to=[], logging_steps=200)
    trainer = Trainer(model=model, args=targs, train_dataset=DS(train), eval_dataset=DS(hold),
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    trainer.train()
    result = trainer.evaluate()
    print("aux held-out:", {k: round(v, 4) for k, v in result.items() if "f1" in k or "loss" in k}, flush=True)

    # Save the encoder only: the stage-2 trainer adds a fresh HOT/THAR head on top.
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT / "_full")
    AutoModel.from_pretrained(OUT / "_full").save_pretrained(OUT)
    tok.save_pretrained(OUT)
    (OUT / "aux_report.json").write_text(json.dumps({"rows": len(rows), "by_source": by_src, "removed_duplicates": dropped,
                                                      "aux_holdout": result, "base_model": args.base_model,
                                                      "epochs": args.epochs, "seed": args.seed}, indent=2), encoding="utf-8")
    print("saved encoder to", OUT, flush=True)


if __name__ == "__main__":
    main()
