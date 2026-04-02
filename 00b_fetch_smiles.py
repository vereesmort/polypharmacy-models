"""
00b_fetch_smiles.py

Fetch canonical SMILES for all DECAGON drugs from PubChem.

Fixes over original version:
    - Proper User-Agent header (Colab blocks anonymous urllib agents)
    - Batch API: up to 100 CIDs per request (10x fewer calls)
    - Exponential backoff retry on rate-limit (HTTP 503/429)
    - Incremental progress save — safe to interrupt and resume
    - Single-CID fallback for any batch failures
    - Connectivity test before starting

Outputs:
    data/drug_smiles.json      {stitch_id: smiles_string}
    data/drug_smiles.csv       stitch_id, pubchem_cid, smiles

If PubChem is completely blocked (DNS failure):
    See MANUAL FALLBACK section at the bottom of this file.
    You can also skip structural similarity edges by setting
    USE_STRUCTURAL_SIM = False in 02b_build_expanded_graph.py.
"""

import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

RAW           = Path("data/raw")
DATA          = Path("data")
OUT_JSON      = DATA / "drug_smiles.json"
OUT_CSV       = DATA / "drug_smiles.csv"
PROGRESS_FILE = DATA / "drug_smiles_progress.json"

BATCH_SIZE    = 100     # PubChem allows up to 100 CIDs per batch
SLEEP_BATCH   = 0.4     # seconds between batch requests
SLEEP_SINGLE  = 0.25    # seconds between single fallback requests
MAX_RETRIES   = 3
RETRY_WAIT    = 5       # seconds, doubled on each retry


# ── Helpers ───────────────────────────────────────────────────────────────────

def stitch_to_cid(stitch_id: str) -> int:
    try:
        return int(stitch_id.replace("CID", "").lstrip("0") or "0")
    except ValueError:
        return 0


def make_request(url: str) -> bytes | None:
    """Fetch URL with proper headers and retry on rate-limit."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < MAX_RETRIES:
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"    Rate limit (HTTP {e.code}) — waiting {wait}s...")
                time.sleep(wait)
            else:
                return None
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                return None
    return None


def fetch_batch(cids: list) -> dict:
    """Fetch SMILES for up to 100 CIDs in one request."""
    cid_str = ",".join(str(c) for c in cids)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid_str}/property/CanonicalSMILES/JSON"
    )
    raw = make_request(url)
    if not raw:
        return {}
    try:
        data  = json.loads(raw)
        props = data.get("PropertyTable", {}).get("Properties", [])
        return {int(p["CID"]): p["CanonicalSMILES"] for p in props
                if "CanonicalSMILES" in p}
    except Exception:
        return {}


def fetch_single(cid: int) -> str | None:
    """Fetch SMILES for one CID."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid}/property/CanonicalSMILES/JSON"
    )
    raw = make_request(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        return None


def check_connectivity() -> bool:
    """Test PubChem with aspirin (CID 2244)."""
    print("Testing PubChem connectivity (aspirin CID 2244)...")
    result = fetch_single(2244)
    if result:
        print(f"  OK — accessible. Aspirin: {result[:50]}")
        return True
    print("  FAILED — PubChem not reachable.")
    print()
    print("  Most likely causes in Colab:")
    print("  1. Runtime has no internet — go to Runtime > Disconnect and reconnect")
    print("  2. Try: !curl -s 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/CanonicalSMILES/JSON'")
    print("  3. DNS test: !nslookup pubchem.ncbi.nlm.nih.gov")
    print()
    print("  If blocked, see MANUAL FALLBACK at the bottom of this script.")
    return False


def load_all_drug_ids() -> list:
    """Collect all unique STITCH drug IDs from raw DECAGON files."""
    drugs = set()
    sources = list(RAW.glob("bio-decagon-*.csv"))
    for fpath in sources:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in (reader.fieldnames or []):
                    if "STITCH" in col.upper():
                        v = row[col].strip()
                        if v.startswith("CID"):
                            drugs.add(v)
    return sorted(drugs)


def save_outputs(smiles: dict, all_drugs: list):
    with open(OUT_JSON, "w") as f:
        json.dump(smiles, f, indent=2)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug_id", "pubchem_cid", "smiles"])
        for drug_id in all_drugs:
            cid = stitch_to_cid(drug_id)
            w.writerow([drug_id, cid, smiles.get(drug_id, "")])
    print(f"Saved → {OUT_JSON}")
    print(f"Saved → {OUT_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA.mkdir(exist_ok=True)

    # Resume from interrupted run
    smiles = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            smiles = json.load(f)
        print(f"Resuming: {len(smiles)} already fetched from previous run.")

    all_drugs = load_all_drug_ids()
    remaining = [d for d in all_drugs if d not in smiles]

    print(f"Total drugs: {len(all_drugs)}")
    print(f"Already fetched: {len(smiles)}")
    print(f"To fetch: {len(remaining)}")

    if not remaining:
        print("All drugs fetched — nothing to do.")
        save_outputs(smiles, all_drugs)
        return

    if not check_connectivity():
        return

    # Build CID <-> STITCH lookup
    cid_to_stitch = {stitch_to_cid(d): d for d in remaining if stitch_to_cid(d) > 0}
    valid_cids    = sorted(cid_to_stitch.keys())

    print(f"\nFetching {len(valid_cids)} drugs via batch API...")
    failed_cids = []

    for i in range(0, len(valid_cids), BATCH_SIZE):
        batch        = valid_cids[i : i + BATCH_SIZE]
        batch_result = fetch_batch(batch)

        for cid in batch:
            stitch = cid_to_stitch[cid]
            if cid in batch_result:
                smiles[stitch] = batch_result[cid]
            else:
                failed_cids.append(cid)

        done = min(i + BATCH_SIZE, len(valid_cids))
        n_ok = len(smiles)
        print(f"  {done}/{len(valid_cids)}  "
              f"(fetched: {n_ok}, failed so far: {len(failed_cids)})")

        # Incremental save every batch
        with open(PROGRESS_FILE, "w") as f:
            json.dump(smiles, f)

        time.sleep(SLEEP_BATCH)

    # Single-CID fallback for failed batches
    if failed_cids:
        print(f"\nFallback: retrying {len(failed_cids)} failed CIDs individually...")
        still_failed = []
        for j, cid in enumerate(failed_cids):
            result = fetch_single(cid)
            stitch = cid_to_stitch[cid]
            if result:
                smiles[stitch] = result
            else:
                still_failed.append(stitch)
            if (j + 1) % 20 == 0:
                print(f"  {j+1}/{len(failed_cids)}")
            time.sleep(SLEEP_SINGLE)

        if still_failed:
            print(f"\n  Could not fetch {len(still_failed)} drugs:")
            for s in still_failed[:15]:
                print(f"    {s} (CID {stitch_to_cid(s)})")
            if len(still_failed) > 15:
                print(f"    ... and {len(still_failed)-15} more")

    # Final save
    save_outputs(smiles, all_drugs)
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    print(f"\nResult: {len(smiles)}/{len(all_drugs)} drugs "
          f"({len(smiles)/len(all_drugs)*100:.1f}% coverage)")


if __name__ == "__main__":
    main()


# ═════════════════════════════════════════════════════════════════════════════
# MANUAL FALLBACK — if PubChem is completely inaccessible from Colab
# ═════════════════════════════════════════════════════════════════════════════
#
# QUICK TEST in Colab first:
#   !curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/CanonicalSMILES/JSON"
#   !nslookup pubchem.ncbi.nlm.nih.gov
#
# If those work, the issue is with the Python script headers. Try:
#   !pip install requests
#   Then replace urllib with requests in fetch_single() — see Option A below.
#
# OPTION A — Use requests library (better SSL/header handling):
#
#   import requests
#   session = requests.Session()
#   session.headers.update({"User-Agent": "Mozilla/5.0"})
#
#   def fetch_single_requests(cid):
#       url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
#       r = session.get(url, timeout=10)
#       if r.status_code == 200:
#           return r.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
#       return None
#
# OPTION B — Download SMILES on your local machine, upload to Colab:
#
#   On your local machine (where internet works):
#     python 00b_fetch_smiles.py
#   Then upload data/drug_smiles.json to your Colab session.
#
# OPTION C — Skip structural similarity (fastest workaround):
#
#   In 02b_build_expanded_graph.py, change:
#     USE_STRUCTURAL_SIM = True
#   to:
#     USE_STRUCTURAL_SIM = False
#
#   The model will train without drug->drug structural edges.
#   This is a valid ablation and you can note it in your thesis.
#   You can always add structural similarity later once you have SMILES.
#
# OPTION D — Use a pre-built SMILES dataset:
#
#   Several databases provide bulk SMILES downloads:
#   - DrugBank approved drugs: https://go.drugbank.com/releases/latest#open-data
#   - ChEMBL: https://www.ebi.ac.uk/chembl/  (search by CID)
#   - PubChem FTP bulk: https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/
#
#   After downloading, convert to the required format:
#   {
#     "CID000002244": "CC(=O)Oc1ccccc1C(=O)O",   <- aspirin
#     "CID000003440": "...",
#     ...
#   }
#   Save as data/drug_smiles.json
