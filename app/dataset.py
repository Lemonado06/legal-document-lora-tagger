"""Milestone 1: build / load a dataset of legal clauses with a tag per clause.

Three sources:
  * synthetic  - offline, generated from templates below (good for smoke tests
                 and for checking the pipeline works; accuracy will be optimistic)
  * ledgar     - real contract provisions from LEDGAR (LexGLUE), pulled with
                 the `datasets` library. This is the one to report numbers on.
  * csv/jsonl  - your own file with `text` and `label` columns
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- synthetic --

PARTIES = [
    ("the Company", "the Contractor"), ("Licensor", "Licensee"), ("the Seller", "the Buyer"),
    ("the Landlord", "the Tenant"), ("the Provider", "the Customer"), ("the Employer", "the Employee"),
    ("the Disclosing Party", "the Receiving Party"), ("the Vendor", "the Client"),
]
STATES = ["Karnataka", "Maharashtra", "Delaware", "New York", "California", "England and Wales",
          "Singapore", "Ontario", "Texas", "Tamil Nadu"]
CITIES = ["Bengaluru", "Mumbai", "Wilmington", "London", "Singapore", "Chennai", "New Delhi", "Toronto"]
DAYS = ["ten (10)", "fifteen (15)", "thirty (30)", "forty-five (45)", "sixty (60)", "ninety (90)"]
YEARS = ["one (1)", "two (2)", "three (3)", "five (5)", "seven (7)"]
AMOUNTS = ["the total fees paid in the twelve (12) months preceding the claim",
           "INR 10,00,000", "USD 50,000", "the amount paid under the applicable order",
           "two times the annual contract value"]
FORUMS = ["the Indian Council of Arbitration", "the Singapore International Arbitration Centre",
          "the American Arbitration Association", "the London Court of International Arbitration"]

TEMPLATES: dict[str, list[str]] = {
    "governing_law": [
        "This Agreement shall be governed by and construed in accordance with the laws of {state}, without regard to its conflict of laws principles.",
        "The validity, interpretation and performance of this Agreement are governed by the laws of {state}.",
        "All matters arising out of or relating to this Agreement shall be governed by the internal laws of {state}.",
        "This contract and any non-contractual obligations connected with it are subject to the laws of {state}.",
        "The parties agree that the substantive law of {state} applies to this Agreement, excluding the UN Convention on Contracts for the International Sale of Goods.",
    ],
    "confidentiality": [
        "{b} shall keep all Confidential Information of {a} strictly confidential and shall not disclose it to any third party without prior written consent.",
        "Each party agrees to use the other party's Confidential Information solely for the purpose of performing this Agreement and to protect it with at least reasonable care.",
        "{b} shall not, during the term and for {years} years thereafter, disclose any non-public information received from {a}.",
        "Confidential Information excludes information that is or becomes publicly available through no fault of {b}.",
        "Upon request, {b} shall promptly return or destroy all documents containing Confidential Information of {a}.",
    ],
    "termination": [
        "Either party may terminate this Agreement upon {days} days' prior written notice to the other party.",
        "{a} may terminate this Agreement immediately if {b} commits a material breach and fails to cure it within {days} days of notice.",
        "This Agreement may be terminated by either party if the other becomes insolvent or makes an assignment for the benefit of creditors.",
        "Upon expiry or termination of this Agreement, all licences granted hereunder shall cease and {b} shall stop using the deliverables.",
        "{b} may terminate for convenience by giving {days} days' written notice, subject to payment of fees accrued up to the termination date.",
    ],
    "indemnification": [
        "{b} shall indemnify, defend and hold harmless {a} from any claims, losses and expenses arising out of {b}'s negligence or wilful misconduct.",
        "Each party shall indemnify the other against third-party claims resulting from its breach of this Agreement.",
        "{a} agrees to defend {b} against any claim that the deliverables infringe a third party's intellectual property rights and pay any resulting damages.",
        "The indemnified party shall give prompt notice of any claim and allow the indemnifying party to control its defence and settlement.",
        "{b} will reimburse {a} for all reasonable attorneys' fees incurred in connection with any indemnified claim.",
    ],
    "limitation_of_liability": [
        "In no event shall either party be liable for any indirect, incidental, special or consequential damages, including loss of profits.",
        "The aggregate liability of {a} under this Agreement shall not exceed {amount}.",
        "Neither party's total liability arising out of this Agreement, whether in contract, tort or otherwise, shall exceed {amount}.",
        "Nothing in this Agreement limits liability for death or personal injury caused by negligence, or for fraud.",
        "{a} shall not be responsible for any loss of data, business interruption or lost revenue, however caused.",
    ],
    "payment": [
        "{b} shall pay all invoices within {days} days of the invoice date.",
        "Fees are exclusive of applicable taxes, which shall be borne by {b}.",
        "Late payments shall accrue interest at the rate of one and a half percent (1.5%) per month until paid in full.",
        "{a} shall invoice {b} monthly in arrears for the services performed during the preceding month.",
        "All amounts payable under this Agreement shall be paid by bank transfer to the account designated by {a}.",
    ],
    "force_majeure": [
        "Neither party shall be liable for any delay or failure to perform caused by events beyond its reasonable control, including acts of God, flood, war or epidemic.",
        "If a Force Majeure Event continues for more than {days} days, either party may terminate this Agreement by written notice.",
        "The affected party shall promptly notify the other of the force majeure event and use reasonable efforts to mitigate its effects.",
        "Performance obligations shall be suspended for so long as a natural disaster, strike, riot or government action prevents performance.",
    ],
    "assignment": [
        "{b} may not assign or transfer this Agreement without the prior written consent of {a}.",
        "Either party may assign this Agreement to a successor in connection with a merger, acquisition or sale of substantially all of its assets.",
        "Any purported assignment in violation of this clause shall be null and void.",
        "This Agreement shall be binding upon and inure to the benefit of the parties and their permitted successors and assigns.",
        "{a} may subcontract or delegate its obligations to an affiliate, provided it remains responsible for their performance.",
    ],
    "dispute_resolution": [
        "Any dispute arising out of this Agreement shall be finally resolved by arbitration administered by {forum}, seated in {city}.",
        "The parties shall first attempt in good faith to resolve any dispute through negotiation between senior executives for a period of {days} days.",
        "The courts of {city} shall have exclusive jurisdiction over any dispute arising under this Agreement.",
        "Disputes shall be referred to a sole arbitrator appointed by mutual agreement, and the award shall be final and binding.",
        "Before commencing litigation, the parties shall submit the dispute to mediation in {city}.",
    ],
    "warranties": [
        "{a} warrants that the services will be performed in a professional and workmanlike manner consistent with industry standards.",
        "Each party represents and warrants that it has full power and authority to enter into this Agreement.",
        "EXCEPT AS EXPRESSLY SET OUT HEREIN, THE SOFTWARE IS PROVIDED 'AS IS' AND {a} DISCLAIMS ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.",
        "{a} warrants that the goods will be free from defects in materials and workmanship for {years} years from delivery.",
        "{b} represents that the information provided to {a} is accurate and complete in all material respects.",
    ],
}

LABELS = list(TEMPLATES.keys())

_PREFIXES = ["", "", "", "Notwithstanding anything to the contrary, ", "Subject to the terms hereof, ",
             "For the avoidance of doubt, ", "Save as otherwise provided, "]
_SUFFIXES = ["", "", "", " This clause shall survive termination of this Agreement.",
             " The parties acknowledge that this provision has been specifically negotiated.",
             " This obligation applies to each party's affiliates as well."]


def _fill(template: str, rng: random.Random) -> str:
    a, b = rng.choice(PARTIES)
    text = template.format(a=a, b=b, state=rng.choice(STATES), city=rng.choice(CITIES),
                           days=rng.choice(DAYS), years=rng.choice(YEARS),
                           amount=rng.choice(AMOUNTS), forum=rng.choice(FORUMS))
    prefix = rng.choice(_PREFIXES)
    if prefix:
        text = prefix + text[0].lower() + text[1:] if not text[:4].isupper() else prefix + text
    text = text + rng.choice(_SUFFIXES)
    return text[0].upper() + text[1:]


def make_synthetic(n_per_label: int = 60, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    rows, seen = [], set()
    for label, templates in TEMPLATES.items():
        made, tries = 0, 0
        while made < n_per_label and tries < n_per_label * 50:
            tries += 1
            text = _fill(rng.choice(templates), rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": label})
            made += 1
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------- ledgar --

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_ledgar(top_k: int = 10, per_label: int = 300, seed: int = 13) -> list[dict]:
    """Real provisions from LEDGAR. Keeps the `top_k` most frequent labels and
    samples up to `per_label` examples of each so training stays laptop-sized."""
    from datasets import load_dataset  # lazy: only needed for this source

    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    pooled = []
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            pooled.append({"text": ex["text"], "label": _slug(names[ex["label"]])})
    keep = {lab for lab, _ in Counter(r["label"] for r in pooled).most_common(top_k)}
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in pooled:
        if r["label"] in keep:
            by[r["label"]].append(r)
    out = []
    for lab, items in by.items():
        rng.shuffle(items)
        out.extend(items[:per_label])
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------ helpers --

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"unsupported file type: {path.suffix} (use .csv or .jsonl)")
    out = []
    for r in rows:
        text, label = clean_text(r.get("text", "")), _slug(r.get("label", ""))
        if text and label:
            out.append({"text": text, "label": label})
    return out


def write_jsonl(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def stratified_split(rows: list[dict], val: float = 0.1, test: float = 0.15, seed: int = 13):
    """Per-label split so rare tags still show up in val and test."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    train, dev, tst = [], [], []
    for lab in sorted(by):
        items = by[lab][:]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test)) if n >= 3 else 0
        n_val = max(1, round(n * val)) if n >= 3 else 0
        tst += items[:n_test]
        dev += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    for part in (train, dev, tst):
        rng.shuffle(part)
    return train, dev, tst


def split_document(text: str, min_chars: int = 25) -> list[str]:
    """Break a contract into clauses: numbered sections first, blank lines second."""
    text = text.replace("\r\n", "\n")
    parts = re.split(r"\n\s*\n|\n(?=\s*(?:\d+(?:\.\d+)*[.)]|\([a-z0-9]+\)|Section \d+|Article [IVX\d]+)\s)", text)
    clauses = []
    for p in parts:
        p = clean_text(p)
        p = re.sub(r"^(?:\d+(?:\.\d+)*[.)]|\([a-z0-9]+\)|Section \d+[.:]?|Article [IVX\d]+[.:]?)\s*", "", p)
        is_heading = p.upper() == p and len(p.split()) <= 8
        if len(p) >= min_chars and not is_heading:
            clauses.append(p)
    return clauses
