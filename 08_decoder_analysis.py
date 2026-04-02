"""
08_decoder_analysis.py
Extract DEDICOM decoder D_r weights from the trained checkpoint and
produce two visualisations:

1. Heatmap  — |D_r[dim]| for the top-15 most discriminative dimensions
              across all 30 SE types  (saved as SVG)

2. Bar chart — top-weighted dimension per SE type, coloured by
              pharmacological group  (saved as SVG)

Outputs:
    results/decoder_D_weights.npy          — raw D matrix [num_se, hidden_dim]
    results/decoder_R_weights.npy          — raw R matrix [hidden_dim, hidden_dim]
    results/decoder_dim_heatmap.svg
    results/decoder_top_dim_per_se.svg
    results/decoder_dim_groups.json        — dimension → SE cluster mapping

Usage:
    python 08_decoder_analysis.py

Requires: numpy, matplotlib (pip install matplotlib)
The checkpoint is read WITHOUT torch using the ZIP/pickle approach,
so this script runs even if PyTorch is not installed locally.
"""

import csv
import json
import struct
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

CHECKPOINT   = Path("checkpoints/best_model.pt")
RESULTS      = Path("results")
RESULTS.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
RAW          = Path("data/raw")
CATEGORIES_CSV = RAW / "bio-decagon-effectcategories.csv"
TOP_N_SE     = 30   # must match training

# Pharmacological group colours (hex) — edit to taste
DIM_GROUPS = {
    15: {"label": "CNS / cardiovascular",    "color": "#534AB7",
         "ses":  ["Chest pain", "Dizziness", "High blood pressure", "Anxiety", "Aching joints"]},
    16: {"label": "Renal toxicity",           "color": "#D85A30",
         "ses":  ["kidney failure", "acute kidney failure"]},
    55: {"label": "Haematological / infectious", "color": "#1D9E75",
         "ses":  ["anaemia", "neumonia"]},
    41: {"label": "Circulatory / GI",         "color": "#BA7517",
         "ses":  ["arterial pressure NOS decreased", "nausea", "emesis"]},
    10: {"label": "Respiratory / musculoskeletal", "color": "#185FA5",
         "ses":  ["Hypoventilation", "Back Ache"]},
     5: {"label": "Metabolic",                "color": "#993556",
         "ses":  ["asthenia", "dehydration", "hyperglycaemia"]},
    57: {"label": "Systemic / constitutional","color": "#5F5E5A",
         "ses":  ["Fatigue", "edema extremities", "loss of weight"]},
}
DEFAULT_COLOR = "#888780"


# ── 1. Extract D and R from checkpoint (no torch required) ─────────────────────

def find_storage_key(pkl_bytes: bytes, param_name: str) -> str:
    """
    Scan the pickle byte stream for the string matching param_name,
    then extract the immediately following storage-key string.
    Returns the key as a string (e.g. '72').
    """
    target = param_name.encode()
    offset = pkl_bytes.find(target)
    if offset == -1:
        raise ValueError(f"Parameter '{param_name}' not found in checkpoint pickle.")
    # Scan forward for the next SHORT_BINUNICODE (0x58) string of length 1-4
    i = offset + len(target)
    end = min(i + 120, len(pkl_bytes))
    while i < end:
        if pkl_bytes[i] == 0x58:
            length = struct.unpack("<I", pkl_bytes[i+1:i+5])[0]
            if 1 <= length <= 4:
                key = pkl_bytes[i+5:i+5+length].decode("utf-8", errors="ignore")
                if key.isdigit():
                    return key
            i += 5 + length
        else:
            i += 1
    raise ValueError(f"Could not find storage key for '{param_name}'.")


def load_tensor(zf: zipfile.ZipFile, storage_key: str,
                shape: tuple, dtype=np.float32) -> np.ndarray:
    path = f"best_model/data/{storage_key}"
    with zf.open(path) as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(shape)


def extract_decoder_weights(checkpoint_path: Path, num_se: int, hidden_dim: int):
    with zipfile.ZipFile(checkpoint_path) as zf:
        with zf.open("best_model/data.pkl") as f:
            pkl = f.read()

        r_key = find_storage_key(pkl, "decoder.R")
        d_key = find_storage_key(pkl, "decoder.D")
        print(f"  decoder.R → storage/{r_key}   decoder.D → storage/{d_key}")

        R = load_tensor(zf, r_key, (hidden_dim, hidden_dim))
        D = load_tensor(zf, d_key, (num_se, hidden_dim))

    return R, D


# ── 2. Load SE metadata ────────────────────────────────────────────────────────

def load_top_se(n: int):
    se_counts = Counter()
    se_names  = {}
    for chunk in sorted(Path("data/raw").glob("bio-decagon-combo*.csv")):
        with open(chunk) as f:
            for row in csv.DictReader(f):
                se_counts[row["Polypharmacy Side Effect"]] += 1
                se_names[row["Polypharmacy Side Effect"]] = row["Side Effect Name"]
    top = [se for se, _ in se_counts.most_common(n)]
    return top, {se: se_names[se] for se in top}


def load_categories():
    cats = {}
    if not CATEGORIES_CSV.exists():
        return cats
    with open(CATEGORIES_CSV) as f:
        for row in csv.DictReader(f):
            cats[row["Side Effect"]] = row["Disease Class"]
    return cats


# ── 3. Dimension group colour lookup ──────────────────────────────────────────

def dim_color(top_dim: int) -> str:
    return DIM_GROUPS.get(top_dim, {}).get("color", DEFAULT_COLOR)


# ── 4. SVG helpers ─────────────────────────────────────────────────────────────

def svg_heatmap(D: np.ndarray, se_names: list, top_dims: list,
                out_path: Path):
    """
    Heatmap: rows = SE types, cols = top discriminative dims.
    Each cell is colour-scaled within its column (min→max of |D[:,dim]|).
    """
    D_abs  = np.abs(D)
    nRows  = len(se_names)
    nCols  = len(top_dims)
    cellW, cellH = 36, 20
    leftPad, topPad = 195, 28
    W = leftPad + nCols * cellW + 80
    H = topPad  + nRows * cellH + 20

    col_min = [D_abs[:, d].min() for d in top_dims]
    col_max = [D_abs[:, d].max() for d in top_dims]

    def blue(t):
        r = int(230 - t * 188)
        g = int(241 - t * 197)
        b = int(251 - t * 198)
        return f"rgb({r},{g},{b})"

    def txt_col(t):
        return "#ffffff" if t > 0.55 else "#0C447C"

    lines = [
        f'<svg width="800" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        '<style>.rl{font-size:10px;fill:#5F5E5A;text-anchor:end;dominant-baseline:central}'
        '.ch{font-size:10px;font-weight:500;text-anchor:middle;fill:#888780}'
        '.cv{font-size:9px;text-anchor:middle;dominant-baseline:central}</style>',
    ]

    # Column headers
    for j, dim in enumerate(top_dims):
        x = leftPad + j * cellW + cellW // 2
        lines.append(f'<text class="ch" x="{x}" y="{topPad - 6}">d{dim}</text>')

    # Cells
    for i, name in enumerate(se_names):
        y = topPad + i * cellH
        lines.append(f'<text class="rl" x="{leftPad - 3}" y="{y + cellH // 2}">'
                     f'{name[:28]}</text>')
        for j, dim in enumerate(top_dims):
            val = float(D_abs[i, dim])
            mn, mx = col_min[j], col_max[j]
            t   = (val - mn) / (mx - mn + 1e-8)
            x   = leftPad + j * cellW
            fill = blue(t)
            tc   = txt_col(t)
            lines.append(
                f'<rect x="{x+1}" y="{y+1}" width="{cellW-2}" height="{cellH-2}" '
                f'rx="2" fill="{fill}"/>'
                f'<text class="cv" x="{x + cellW//2}" y="{y + cellH//2}" '
                f'fill="{tc}">{val:.2f}</text>'
            )

    # Highlight kinase dim column (dim 16 = renal)
    for highlight_dim, hl_color in [(16, "#D85A30"), (15, "#534AB7"), (55, "#1D9E75")]:
        if highlight_dim in top_dims:
            j = top_dims.index(highlight_dim)
            x = leftPad + j * cellW
            lines.append(
                f'<rect x="{x}" y="{topPad}" width="{cellW}" '
                f'height="{nRows * cellH}" rx="0" fill="none" '
                f'stroke="{hl_color}" stroke-width="1.5"/>'
            )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines))
    print(f"  Saved heatmap → {out_path}")


def svg_bar_chart(D: np.ndarray, se_names: list, out_path: Path):
    """
    Horizontal bar chart: one bar per SE type, length = max |D_r[dim]|,
    coloured by the pharmacological group of the top dimension.
    """
    D_abs = np.abs(D)
    nRows = len(se_names)

    row_height = 22
    leftPad    = 195
    barMaxW    = 340    # width for weight = 1.0
    rightPad   = 90
    topPad     = 28
    H = topPad + nRows * row_height + 20
    W = leftPad + barMaxW + rightPad

    lines = [
        f'<svg width="680" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        '<style>'
        '.rl{font-size:10px;fill:#5F5E5A;text-anchor:end;dominant-baseline:central}'
        '.dl{font-size:9px;fill:#5F5E5A;text-anchor:start;dominant-baseline:central}'
        '.ax{font-size:9px;fill:#B4B2A9;text-anchor:middle}'
        '</style>',
        f'<text class="ax" x="{leftPad + barMaxW//2}" y="14">'
        f'|D_r| decoder weight (0 = unused → 1 = maximum importance)</text>',
    ]

    # Grid lines
    for val, label in [(0.0, "0.0"), (0.3, "0.3"), (0.6, "0.6"), (0.9, "0.9")]:
        gx = leftPad + int(val * barMaxW)
        lines.append(
            f'<line x1="{gx}" y1="{topPad}" x2="{gx}" y2="{topPad + nRows*row_height}" '
            f'stroke="#E8E6DE" stroke-width="0.5"/>'
            f'<text class="ax" x="{gx}" y="{topPad + nRows*row_height + 12}">{label}</text>'
        )

    # Bars
    for i, name in enumerate(se_names):
        weights    = D_abs[i]
        top_dim    = int(np.argmax(weights))
        top_weight = float(weights[top_dim])
        bar_w      = int(top_weight * barMaxW)
        color      = dim_color(top_dim)
        y          = topPad + i * row_height
        cy         = y + row_height // 2

        lines.append(
            f'<text class="rl" x="{leftPad - 3}" y="{cy}">{name[:26]}</text>'
            f'<rect x="{leftPad}" y="{y + 4}" width="{bar_w}" height="{row_height - 8}" '
            f'rx="2" fill="{color}"/>'
            f'<text class="dl" x="{leftPad + bar_w + 4}" y="{cy}">'
            f'd{top_dim}: {top_weight:.2f}</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines))
    print(f"  Saved bar chart → {out_path}")


# ── 5. Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Infer model config from checkpoint ────────────────────────────────────
    # We need num_se and hidden_dim to know the tensor shapes.
    # Try to infer from the zip file sizes.
    with zipfile.ZipFile(CHECKPOINT) as zf:
        with zf.open("best_model/data.pkl") as f:
            pkl = f.read()
        # Find the storage key for decoder.D and read its size
        try:
            d_key = find_storage_key(pkl, "decoder.D")
            r_key = find_storage_key(pkl, "decoder.R")
        except ValueError as e:
            print(f"Error: {e}")
            return
        d_size = zf.getinfo(f"best_model/data/{d_key}").file_size
        r_size = zf.getinfo(f"best_model/data/{r_key}").file_size

    n_d_floats = d_size // 4   # float32
    n_r_floats = r_size // 4
    # R is square: hidden_dim = sqrt(n_r_floats)
    hidden_dim = int(round(n_r_floats ** 0.5))
    num_se     = n_d_floats // hidden_dim
    print(f"Inferred: hidden_dim={hidden_dim}, num_se={num_se}")

    # ── Extract weights ───────────────────────────────────────────────────────
    print("Extracting decoder weights from checkpoint...")
    R, D = extract_decoder_weights(CHECKPOINT, num_se, hidden_dim)
    np.save(RESULTS / "decoder_D_weights.npy", D)
    np.save(RESULTS / "decoder_R_weights.npy", R)
    print(f"  D shape: {D.shape}   R shape: {R.shape}")

    # ── Load SE names ─────────────────────────────────────────────────────────
    print("Loading SE metadata...")
    top_se_ids, top_se_names_map = load_top_se(num_se)
    se_names = [top_se_names_map[se] for se in top_se_ids]
    cats     = load_categories()
    se_cats  = [cats.get(se, "unannotated") for se in top_se_ids]

    # ── Top discriminative dimensions ─────────────────────────────────────────
    D_abs         = np.abs(D)
    var_per_dim   = D_abs.var(axis=0)
    top15_dims    = np.argsort(var_per_dim)[::-1][:15].tolist()
    print(f"Top 15 discriminative dims: {top15_dims}")

    # ── Dimension groups ──────────────────────────────────────────────────────
    # For each SE, record which dim it routes through primarily
    primary_dim_map = {}
    for i, name in enumerate(se_names):
        top_dim = int(np.argmax(D_abs[i]))
        group   = DIM_GROUPS.get(top_dim, {}).get("label", "other")
        primary_dim_map[name] = {"top_dim": top_dim, "weight": float(D_abs[i, top_dim]),
                                 "group": group}

    with open(RESULTS / "decoder_dim_groups.json", "w") as f:
        json.dump(primary_dim_map, f, indent=2)
    print(f"  Saved dim groups → {RESULTS}/decoder_dim_groups.json")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nTop decoder dimension per SE type:")
    print(f"  {'SE':<40} {'Top dim':>8}  {'Weight':>7}  Group")
    print("  " + "-"*72)
    for i, name in enumerate(se_names):
        top_dim = int(np.argmax(D_abs[i]))
        weight  = float(D_abs[i, top_dim])
        group   = DIM_GROUPS.get(top_dim, {}).get("label", "—")
        print(f"  {name[:38]:<40} {'d'+str(top_dim):>8}  {weight:>7.3f}  {group}")

    print("\nMost discriminative dims (variance across SEs):")
    for dim in top15_dims[:10]:
        v = float(var_per_dim[dim])
        mn, mx = float(D_abs[:,dim].min()), float(D_abs[:,dim].max())
        print(f"  d{dim:2d}  var={v:.5f}  range=[{mn:.3f}, {mx:.3f}]")

    # ── SVG outputs ───────────────────────────────────────────────────────────
    print("\nGenerating SVG charts...")
    svg_heatmap(D, se_names, top15_dims, RESULTS / "decoder_dim_heatmap.svg")
    svg_bar_chart(D, se_names,           RESULTS / "decoder_top_dim_per_se.svg")

    print("\nDone. Open results/ to view the SVG files.")
    print("You can embed them directly in LaTeX with \\includegraphics{...svg}")
    print("or convert to PDF with: inkscape --export-pdf=chart.pdf chart.svg")


if __name__ == "__main__":
    main()
