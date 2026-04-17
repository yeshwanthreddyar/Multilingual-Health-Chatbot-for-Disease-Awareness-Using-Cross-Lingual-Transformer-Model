from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


NLLB_MODEL = "facebook/nllb-200-distilled-600M"

# Project language codes -> NLLB codes
LANG_MAP: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
    "kok": "gom_Deva",  # Konkani (Goan)
}

INDIAN_LANGS_13 = ["hi", "bn", "te", "ta", "kn", "ml", "mr", "gu", "pa", "or", "as", "ur", "ne"]


def _load_seed_english(symptom2disease_path: str, pubmedqa_path: str) -> pd.DataFrame:
    """
    Build an English seed set from the existing loaders output without requiring IndicNLG.
    Returns a dataframe with at least: text, language, intent, disease, diseases, answer, source.
    """
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from app.data.load_symptom2disease import load_symptom2disease
    from app.data.load_pubmedqa import load_pubmedqa

    rows: List[dict] = []

    s2d = load_symptom2disease(symptom2disease_path)
    for _, r in s2d.iterrows():
        rows.append(
            {
                "text": str(r.get("text", "")).strip(),
                "language": "en",
                "intent": str(r.get("intent", "symptom_reporting")),
                "symptoms": r.get("symptoms", []),
                "disease": str(r.get("disease", "")).strip(),
                "diseases": [],
                "answer": "",
                "source": "symptom2disease",
            }
        )

    pq = load_pubmedqa(pubmedqa_path)
    for _, r in pq.iterrows():
        rows.append(
            {
                "text": str(r.get("text", "")).strip(),
                "language": "en",
                "intent": str(r.get("intent", "disease_information")),
                "symptoms": r.get("symptoms", []),
                "disease": "",
                "diseases": r.get("diseases", []),
                "answer": str(r.get("answer", "")).strip(),
                "source": "pubmedqa",
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 3].reset_index(drop=True)
    return df


def _translator():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def _translate(tokenizer, model, device: str, text: str, tgt_lang_nllb: str) -> str:
    """
    NLLB translation from English to target language.
    """
    tokenizer.src_lang = LANG_MAP["en"]
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_nllb)
    generated = model.generate(**encoded, forced_bos_token_id=forced_bos_token_id, max_new_tokens=128)
    out = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return (out[0] if out else "").strip()


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["text", "language", "intent", "symptoms", "disease", "diseases", "answer", "split", "source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            # store lists as JSON strings so loader can parse them
            rr = dict(r)
            for k in ("symptoms", "diseases"):
                v = rr.get(k, [])
                if isinstance(v, (list, tuple, dict)):
                    rr[k] = json.dumps(v, ensure_ascii=False)
            w.writerow({c: rr.get(c, "") for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symptom2disease", default="Symptom2Disease.csv")
    ap.add_argument("--pubmedqa", default="ori_pqau.json")
    ap.add_argument("--out", default="data/custom_multilingual.csv")
    ap.add_argument("--langs", default=",".join(INDIAN_LANGS_13), help="Comma-separated project language codes")
    ap.add_argument("--per_lang", type=int, default=1000, help="How many samples to generate per target language")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("Loading English seed dataset...")
    seed_df = _load_seed_english(args.symptom2disease, args.pubmedqa)
    if seed_df.empty:
        raise RuntimeError("No seed samples found. Check Symptom2Disease and PubMedQA paths.")

    target_langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    for l in target_langs:
        if l not in LANG_MAP:
            raise ValueError(f"Unsupported language code: {l}. Supported: {sorted(LANG_MAP.keys())}")

    # Sample seed rows (we keep intent/labels; translate only the text and answer)
    base_rows = seed_df.to_dict(orient="records")
    print(f"Seed samples: {len(base_rows)}")

    print("Loading translation model (first run will download weights)...")
    tokenizer, model, device = _translator()
    print(f"Translator ready on {device}")

    out_rows: List[dict] = []

    for lang in target_langs:
        n = min(args.per_lang, len(base_rows))
        chosen = random.sample(base_rows, k=n) if len(base_rows) >= n else base_rows
        tgt = LANG_MAP[lang]
        print(f"Translating {len(chosen)} samples to {lang} ({tgt})...")
        for r in chosen:
            text_en = str(r.get("text", "")).strip()
            ans_en = str(r.get("answer", "")).strip()
            text_t = _translate(tokenizer, model, device, text_en, tgt) if text_en else ""
            ans_t = _translate(tokenizer, model, device, ans_en, tgt) if ans_en else ""
            if not text_t or len(text_t) < 3:
                continue
            out_rows.append(
                {
                    "text": text_t,
                    "language": lang,
                    "intent": r.get("intent", "general_health_query"),
                    "symptoms": r.get("symptoms", []),
                    "disease": r.get("disease", ""),
                    "diseases": r.get("diseases", []),
                    "answer": ans_t,
                    "split": "train",
                    "source": f"nllb_translate_{r.get('source', 'seed')}",
                }
            )

    out_path = Path(args.out)
    _write_csv(out_path, out_rows)
    print(f"\nDone. Wrote {len(out_rows)} rows to {out_path}")
    print("Next: run `python train.py` to train on Symptom2Disease + PubMedQA + custom_multilingual.")


if __name__ == "__main__":
    main()

