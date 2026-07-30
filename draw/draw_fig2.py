"""
Figure 2: Knowledge graph construction and statistics
6 subplots: (a) modality comparison, (b) node type distribution,
(c) CPMP heatmap, (d) community size distribution,
(e) placeholder for Neo4j screenshot, (f) entity word cloud
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from pathlib import Path
from wordcloud import WordCloud

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


def _apply_spine_style(ax):
    """Left/bottom: visible with ticks. Right/top: border only, no ticks."""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(top=False, right=False)

# ── Paths ─────────────────────────────────────────────────────
GRAPH_PATH = Path("../data/graphs/final_graph.json")
CACHE_PATH = Path("../visualizations/analysis_results.pkl")
OUTPUT_DIR = Path("../visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 8-category mapping for node types ─────────────────────────
CATEGORY_MAP = {
    "Material": [
        "Material", "MATERIAL", "Material Sample", "Alloying Element",
        "COMPONENT", "Alloy System", "Alloy", "Superalloy",
        "Nickel-based Superalloy", "Chemical Element", "Solute Atoms",
        "Solute Elements",
    ],
    "Property": [
        "PROPERTY", "Material Property", "Mechanical Property",
        "Physical Property", "MEASUREMENT", "Thermal Property",
        "Creep Property", "Fatigue Property", "Oxidation Property",
    ],
    "Processing": [
        "Process Parameter", "CONDITION", "Heat Treatment Process",
        "Manufacturing Process", "Processing", "Ageing Route",
        "Casting Process", "Welding Process",
    ],
    "Microstructure": [
        "Microstructural Feature", "PHASE", "STRUCTURE",
        "Microstructural Phase", "Microstructural Phenomenon",
        "Crystal Structure", "Grain", "Precipitate",
    ],
    "Defect": [
        "Crystallographic Defect", "Crystal Defect", "Defect",
        "Crack", "Failure Mode", "Degradation",
    ],
    "Equipment": [
        "Equipment", "Instrument", "Testing Method",
        "Characterization Technique", "Experimental Setup",
    ],
    "Literature": [
        "Author", "Scientific Study", "Publication",
        "Research Group", "Institution",
    ],
    "Other": [
        "Technical Category", "COMPONENT", "Concept",
        "Application", "Environment",
    ],
}


def _build_type_lookup():
    """Build reverse lookup: fine-grained type -> category."""
    lookup = {}
    for cat, types in CATEGORY_MAP.items():
        for t in types:
            lookup[t] = cat
            lookup[t.upper()] = cat
    return lookup


TYPE_LOOKUP = _build_type_lookup()


def categorize_type(raw_type: str) -> str:
    """Map a raw node type string to one of 8 categories."""
    if raw_type in TYPE_LOOKUP:
        return TYPE_LOOKUP[raw_type]
    raw_upper = raw_type.upper()
    if raw_upper in TYPE_LOOKUP:
        return TYPE_LOOKUP[raw_upper]
    # Keyword-based fallback (order matters — more specific first)
    kw = raw_upper
    # Defect (before material/structure to catch "crack" etc.)
    if any(k in kw for k in ["DEFECT", "CRACK", "FAILURE", "DEGRAD",
                              "VOID", "POROSITY", "CORROSION", "OXIDAT",
                              "DAMAGE", "FRACTURE", "FATIGUE LIFE"]):
        return "Defect"
    # Microstructure (before material to catch "phase", "precipitate")
    if any(k in kw for k in ["MICRO", "PHASE", "STRUCTURE", "GRAIN",
                              "PRECIPITAT", "CRYSTAL STRUCT", "DISLOC",
                              "DEFORMATION", "REGION", "MORPHOLOG",
                              "TEXTURE", "SEGREGAT", "DENDRIT",
                              "SOLIDIF", "RECRYSTALL", "INTERMETALL",
                              "CRYSTALLOGRAPH", "ORIENTATION", "PLANE",
                              "DIRECTION", "INTERFACE", "OXIDE LAYER",
                              "MECHANISM", "STRUCTURAL FEATURE",
                              "FEATURE"]):
        return "Microstructure"
    # Material
    if any(k in kw for k in ["MATERIAL", "ALLOY", "ELEMENT", "SAMPLE",
                              "COMPONENT", "SUPERALLOY", "STEEL",
                              "ALUMINUM", "NICKEL", "TITANIUM", "COBALT",
                              "SUBSTRATE", "COATING", "SOLUTE", "SYSTEM",
                              "COMPOUND", "CHEMICAL", "METAL", "REAGENT",
                              "ION", "SPECIES", "OXIDE"]):
        return "Material"
    # Property
    if any(k in kw for k in ["PROPERTY", "STRENGTH", "HARDNESS", "MODULUS",
                              "MEASUREMENT", "THERMAL", "CREEP", "TENSILE",
                              "YIELD", "DUCTIL", "TOUGHNESS", "DENSITY",
                              "CONDUCTIV", "METRIC", "PERFORMANCE",
                              "RESISTANCE", "STRESS", "STRAIN", "ELASTIC",
                              "QUANTITY", "PHENOMENON", "BEHAVIOR",
                              "RESPONSE", "TREND", "THERMODYNAMIC"]):
        return "Property"
    # Processing
    if any(k in kw for k in ["PROCESS", "TREATMENT", "CONDITION", "PARAMETER",
                              "MANUFACTUR", "CASTING", "WELD", "ANNEAL",
                              "SINTER", "FORGING", "ROLLING", "MACHINING",
                              "ADDITIVE", "SLM", "LASER", "TEMPERATURE",
                              "AGING", "AGEING", "QUENCH", "COOL"]):
        return "Processing"
    # Equipment / Method
    if any(k in kw for k in ["EQUIP", "INSTRUMENT", "TECHNIQUE", "TEST",
                              "METHOD", "ANALYT", "CHARACTERIZ", "MODEL",
                              "SIMULAT", "COMPUTAT", "SOFTWARE", "SEM",
                              "TEM", "XRD", "EBSD", "DFT", "FEM",
                              "ALGORITHM", "MACHINE LEARN", "FRAMEWORK",
                              "THEORETICAL"]):
        return "Equipment"
    # Literature
    if any(k in kw for k in ["AUTHOR", "STUDY", "PUBLICATION", "JOURNAL",
                              "RESEARCH", "PERSON", "ORGANIZ", "INSTITUT",
                              "UNIVERSIT", "LABORATOR", "STANDARD",
                              "BOOK", "REFERENCE", "DATASET", "FIGURE",
                              "VISUALIZATION"]):
        return "Literature"
    return "Other"


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════

def load_graph_nodes():
    """Load only node data from the graph JSON."""
    print("Loading graph nodes (this may take a moment)...")
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data["nodes"]
    print(f"  Loaded {len(nodes)} nodes")
    return nodes


def load_cpmp_results():
    """Load cached CPMP analysis results from ontology.py."""
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    if isinstance(cache, dict) and "results" in cache:
        return cache["results"], cache.get("categorized_nodes", {})
    return cache, {}


# ══════════════════════════════════════════════════════════════
# Subplot (a): Modality comparison — grouped bar chart
# ══════════════════════════════════════════════════════════════

def plot_modality_comparison(ax):
    modalities = ["Text", "Abstract", "Table", "Image"]
    chunk_pct = [56.9, 17.5, 9.2, 16.5]
    triplet_pct = [54.8, 21.8, 12.8, 10.7]

    x = np.arange(len(modalities))
    w = 0.35
    c1, c2 = "#4C72B0", "#DD8452"

    bars1 = ax.bar(x - w / 2, chunk_pct, w, label="Chunk %", color=c1,
                   edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + w / 2, triplet_pct, w, label="Triplet %", color=c2,
                   edgecolor="black", linewidth=0.6)

    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.8,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 68)
    ax.legend(fontsize=18, frameon=False)
    _apply_spine_style(ax)
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════
# Subplot (b): Node type distribution (8 categories)
# ══════════════════════════════════════════════════════════════

def plot_node_type_distribution(ax, nodes):
    cat_counter = Counter()
    for n in nodes:
        raw = n.get("type", "Unknown")
        cat_counter[categorize_type(raw)] += 1

    # Sort by count descending
    cats = cat_counter.most_common()
    labels = [c[0] for c in cats]
    counts = [c[1] for c in cats]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

    # Horizontal bar chart for better label readability
    y = np.arange(len(labels))
    bars = ax.barh(y, pcts, color=colors[:len(labels)],
                   edgecolor="black", linewidth=0.6)

    for b, p in zip(bars, pcts):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                f"{p:.1f}%", ha="left", va="center", fontsize=16)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Percentage (%)")
    ax.set_xlim(0, max(pcts) * 1.25)
    ax.invert_yaxis()
    _apply_spine_style(ax)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════
# Subplot (c): CPMP 5×5 relationship heatmap
# ══════════════════════════════════════════════════════════════

def plot_cpmp_heatmap(ax, results, categorized_nodes):
    categories = sorted(categorized_nodes.keys())
    n = len(categories)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(categories):
        for j, c2 in enumerate(categories):
            key = (c1, c2)
            if key in results:
                matrix[i, j] = results[key]["count"]

    cmap = LinearSegmentedColormap.from_list("cpmp", ["#F7FBFF", "#2171B5"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    cap_labels = [c.capitalize() for c in categories]
    ax.set_xticklabels(cap_labels, rotation=30, ha="right", fontsize=18)
    ax.set_yticklabels(cap_labels, fontsize=18)
    ax.set_xlabel("Target Category")
    ax.set_ylabel("Source Category")

    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontsize=13)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Heatmap keeps all 4 spines, no extra tick suppression needed
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(top=False, right=False)
    ax.text(-0.18, 1.05, "(c)", transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════
# Subplot (d): Community size distribution (log scale)
# ══════════════════════════════════════════════════════════════

def plot_community_distribution(ax, nodes):
    comm_counter = Counter()
    for n in nodes:
        c = n.get("community")
        if c is not None:
            comm_counter[c] += 1

    # Filter communities with >5 nodes, sort descending
    sizes = sorted([v for v in comm_counter.values() if v > 5], reverse=True)
    if not sizes:
        return

    ax.plot(range(1, len(sizes) + 1), sizes, linewidth=2,
            color="#2B579A", alpha=0.85)
    ax.set_xlabel("Community Rank")
    ax.set_ylabel("Community Size")
    ax.set_yscale("log")
    _apply_spine_style(ax)
    ax.text(-0.12, 1.05, "(d)", transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════
# Subplot (e): Placeholder for Neo4j screenshot
# ══════════════════════════════════════════════════════════════

def plot_neo4j_placeholder(ax):
    ax.text(0.5, 0.5, "Neo4j Subgraph\n(screenshot)",
            ha="center", va="center", fontsize=22, color="#888888",
            style="italic", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linestyle("--")
        spine.set_color("#AAAAAA")
    ax.text(-0.12, 1.05, "(e)", transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════
# Subplot (f): High-frequency entity word cloud
# ══════════════════════════════════════════════════════════════

def extract_wordcloud_candidates(nodes, top_n=500):
    """Extract, normalize, and deduplicate entity names for word cloud.
    Saves to CSV for manual review before plotting."""
    name_counter = Counter()
    # Skip generic / non-domain terms
    skip_lower = {
        "unknown", "none", "n/a", "", "et al.", "et al",
        "temperature", "time", "stress", "strain", "results",
        "figure", "table", "data", "method", "analysis",
        "study", "research", "paper", "work", "effect",
        "effects", "properties", "property", "process",
        "structure", "material", "materials", "alloy", "alloys",
        "sample", "samples", "test", "tests", "value", "values",
        "condition", "conditions", "type", "types", "phase",
        "system", "model", "experiment", "experiments",
        "microstructure", "composition", "content", "rate",
        "surface", "size", "strength", "behavior", "response",
        "region", "area", "volume", "weight", "mass", "ratio",
        "number", "range", "level", "degree", "state", "form",
        "group", "case", "part", "point", "line", "base",
        "high", "low", "large", "small", "new", "different",
        "mechanical properties", "high temperature",
    }
    for n in nodes:
        name = n.get("name", "")
        if (name and name.lower() not in skip_lower
                and len(name) > 2
                and not name.startswith("[")
                and not name.startswith("(")
                and not name[0].isdigit()):
            name_counter[name] += 1

    # ── Normalization: merge duplicates ──
    # 1) Case-insensitive merge (keep the most frequent casing)
    case_groups = {}
    for name, cnt in name_counter.items():
        key = name.lower().strip()
        if key not in case_groups or cnt > case_groups[key][1]:
            case_groups[key] = (name, cnt)
        else:
            case_groups[key] = (case_groups[key][0],
                                case_groups[key][1] + cnt)
    merged = {v[0]: v[1] for v in case_groups.values()}

    # 2) Normalize Unicode variants of quotes/primes
    greek_norm = {
        "\u2032": "'",   # prime → apostrophe
        "\u2033": "''",  # double prime
        "\u2019": "'",   # right single quote
        "\u2018": "'",   # left single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
    }
    # First: normalize existing Unicode subscripts to normal digits
    # so we can re-apply subscript conversion uniformly
    sub_to_normal = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    normalized = {}
    for name, cnt in merged.items():
        norm = name
        for old, new in greek_norm.items():
            norm = norm.replace(old, new)
        norm = norm.translate(sub_to_normal)
        norm = norm.strip()
        if norm in normalized:
            normalized[norm] += cnt
        else:
            normalized[norm] = cnt

    # 3) Convert digits in chemical formulas to Unicode subscripts
    #    Only for chemical formulas: element symbol + digits
    #    Skip alloy designations (GH4169, Inconel 718, UNS N07718, etc.)
    import re
    normal_to_sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    # Known alloy/standard patterns that should NOT be subscripted
    _no_sub_patterns = re.compile(
        r'^(GH\d|IN\d|UNS\s|CMSX|Ren[eé]\s|PWA\s|Mar-M|'
        r'Alloy\s|Inconel\s|Hastelloy\s|Nimonic\s|Udimet\s|'
        r'Waspaloy|AM1|SRR|TMS|DD\d|DZ\d|IC\d)',
        re.IGNORECASE
    )

    def _subscript_formula(text):
        """Convert digits in chemical formulas to Unicode subscripts.
        Skips alloy designations and standalone numbers."""
        # Don't touch alloy designations
        if _no_sub_patterns.match(text):
            return text
        # Don't touch if it's just "word + space + number" (Alloy 718)
        if re.match(r'^[A-Za-z]+\s+\d+', text):
            return text
        # Only subscript digits that follow element-like patterns
        # Match: uppercase letter (+ optional lowercase) then digits
        # Use capturing groups instead of variable-width lookbehind
        return re.sub(
            r'([A-Z][a-z]?)(\d+)',
            lambda m: m.group(1) + m.group(2).translate(normal_to_sub),
            text
        )

    subscripted = {}
    for name, cnt in normalized.items():
        new_name = _subscript_formula(name)
        if new_name in subscripted:
            subscripted[new_name] += cnt
        else:
            subscripted[new_name] = cnt

    # 4) Filter out Chinese-only entries and figure references
    filtered = {}
    for name, cnt in subscripted.items():
        # Skip if all non-ASCII (Chinese-only)
        if not any(c.isascii() and c.isalpha() for c in name):
            continue
        # Skip figure/table references
        if re.match(r"^(Fig|Table|Eq)\.", name):
            continue
        filtered[name] = cnt

    # 4) Sort by count, take top_n
    sorted_words = sorted(filtered.items(), key=lambda x: -x[1])[:top_n]

    # Save to CSV for manual review
    csv_path = OUTPUT_DIR / "wordcloud_candidates.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("word,count,keep\n")
        for word, cnt in sorted_words:
            f.write(f'"{word}",{cnt},Y\n')

    print(f"  Saved {len(sorted_words)} candidates to {csv_path}")
    print("  Edit the 'keep' column (Y/N) and re-run to generate final cloud.")
    return csv_path


def load_wordcloud_words():
    """Load manually reviewed word list from CSV.
    Prefers reviewed CSV if it exists, otherwise falls back to raw."""
    reviewed = OUTPUT_DIR / "wordcloud_candidates_reviewed.csv"
    raw = OUTPUT_DIR / "wordcloud_candidates.csv"
    csv_path = reviewed if reviewed.exists() else raw
    print(f"  Loading words from: {csv_path.name}")
    words = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse CSV: "word",count,keep
            parts = line.rsplit(",", 2)
            if len(parts) == 3:
                word = parts[0].strip('"')
                cnt = int(parts[1])
                keep = parts[2].strip().upper()
                if keep == "Y":
                    words[word] = cnt
    return words


# ══════════════════════════════════════════════════════════════
# LLM-based deduplication
# ══════════════════════════════════════════════════════════════

DEDUPE_PROMPT = """\
You are a materials science terminology expert. I have a list of entity names \
extracted from a superalloy knowledge graph, intended for a word cloud figure \
in an academic paper.

Your tasks:
1. **Merge synonyms/near-duplicates**: Group words that refer to the same \
concept. For each group, pick the best representative term (prefer the \
shortest, most standard English form). Sum their counts.
   - Examples: "M₂₃C₆" / "M₂₃C₆ carbide" / "M₂₃C₆ carbides" → keep "M₂₃C₆"
   - "Dislocations" / "dislocations" → keep "Dislocations"
   - "Aluminum" / "Aluminum (Al)" / "Al (Aluminum)" → keep "Aluminum"
   - "TEM" / "Transmission Electron Microscopy (TEM)" / "Transmission electron microscopy" → keep "TEM"
   - "Ni-based superalloy" / "nickel-based superalloy" / "Ni-base superalloy" → keep one

2. **Mark for deletion** (keep=N):
   - Non-domain terms: funding agencies, table/figure references, generic words
   - Terms containing Chinese characters (they cannot render in the font)
   - Overly generic terms not specific to superalloy/materials science

3. **Keep** domain-specific terms: alloy names, phases, elements, carbides, \
mechanical properties, characterization techniques, processing terms, defects.

Input format (one per line): "word",count
Output format: strict JSON with this structure:
```json
{
  "merge_groups": [
    {
      "representative": "M₂₃C₆",
      "members": ["M₂₃C₆", "M₂₃C₆ carbide", "M₂₃C₆ carbides"],
      "total_count": 563
    }
  ],
  "delete": ["National Natural Science Foundation of China", "Table 1", ...],
  "keep_as_is": ["Inconel 718", "Waspaloy", ...]
}
```

IMPORTANT: Every input word must appear in exactly one of: a merge group's \
members list, the delete list, or the keep_as_is list. Do not omit any word.

Here are the words:
{word_list}
"""


def dedupe_with_llm():
    """Call LLM to intelligently merge and filter word cloud candidates."""
    import os
    from openai import OpenAI

    # Read raw candidates
    csv_path = OUTPUT_DIR / "wordcloud_candidates.csv"
    if not csv_path.exists():
        print("ERROR: Run --extract first to generate candidates.")
        return

    raw_words = []
    with open(csv_path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(",", 2)
            if len(parts) >= 2:
                word = parts[0].strip('"')
                cnt = int(parts[1])
                raw_words.append((word, cnt))

    print(f"  Loaded {len(raw_words)} raw candidates")

    # Build word list string for prompt
    word_list_str = "\n".join(f'"{w}",{c}' for w, c in raw_words)
    prompt = DEDUPE_PROMPT.replace("{word_list}", word_list_str)

    # Call Qwen API
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("ERROR: QWEN_API_KEY environment variable not set.")
        return

    print("  Calling Qwen LLM for deduplication...")
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    reply = response.choices[0].message.content
    print(f"  LLM response received ({len(reply)} chars)")

    # Parse JSON from response (handle markdown code blocks)
    import json as json_mod
    json_str = reply
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]

    # Strip thinking tags if present (Qwen3 may include <think>...</think>)
    if "<think>" in json_str:
        # Remove everything between <think> and </think>
        import re
        json_str = re.sub(r"<think>.*?</think>", "", json_str, flags=re.DOTALL)

    result = json_mod.loads(json_str.strip())

    # Build reviewed word list
    reviewed = {}
    word_to_count = dict(raw_words)

    # Process merge groups
    for group in result.get("merge_groups", []):
        rep = group["representative"]
        members = group.get("members", [])
        total = sum(word_to_count.get(m, 0) for m in members)
        if total == 0:
            total = group.get("total_count", 0)
        reviewed[rep] = (total, "Y")

    # Process keep_as_is
    for word in result.get("keep_as_is", []):
        if word not in reviewed:
            cnt = word_to_count.get(word, 0)
            reviewed[word] = (cnt, "Y")

    # Process deletions
    for word in result.get("delete", []):
        cnt = word_to_count.get(word, 0)
        reviewed[word] = (cnt, "N")

    # Sort by count descending
    sorted_reviewed = sorted(reviewed.items(),
                             key=lambda x: (-x[1][0], x[0]))

    # Save reviewed CSV
    out_path = OUTPUT_DIR / "wordcloud_candidates_reviewed.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("word,count,keep\n")
        for word, (cnt, keep) in sorted_reviewed:
            f.write(f'"{word}",{cnt},{keep}\n')

    n_keep = sum(1 for _, (_, k) in sorted_reviewed if k == "Y")
    n_del = sum(1 for _, (_, k) in sorted_reviewed if k == "N")
    print(f"  Saved to {out_path}")
    print(f"  Result: {n_keep} keep, {n_del} delete")
    print("  Please review the CSV, then re-run without flags to plot.")
    return out_path
def plot_wordcloud(ax, words):
    """Plot word cloud from reviewed word dict {word: count}."""
    # Use Calibri for word cloud (supports Unicode subscripts + Greek)
    import matplotlib.font_manager as fm
    calibri_path = fm.findfont(fm.FontProperties(family="Calibri"))

    wc = WordCloud(
        width=800, height=600,
        background_color="white",
        font_path=calibri_path,
        max_words=150,
        colormap="cividis",
        prefer_horizontal=0.75,
        min_font_size=12,
        max_font_size=90,
        relative_scaling=0.5,
    )
    wc.generate_from_frequencies(words)

    ax.imshow(wc, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.05, 1.05, "(f)", transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top")


# ══════════════════════════════════════════════════════════════
# Main: assemble 2×3 figure
# ══════════════════════════════════════════════════════════════

def main():
    import sys
    extract_only = "--extract" in sys.argv
    dedupe_only = "--dedupe" in sys.argv

    # Step 0: Dedupe mode — call LLM and exit
    if dedupe_only:
        print("\n=== LLM-based deduplication ===")
        dedupe_with_llm()
        return

    # Load data
    nodes = load_graph_nodes()
    cpmp_results, cpmp_cats = load_cpmp_results()

    # Step 1: Extract word cloud candidates for manual review
    csv_path = OUTPUT_DIR / "wordcloud_candidates.csv"
    if extract_only or not csv_path.exists():
        print("\n=== Extracting word cloud candidates ===")
        extract_wordcloud_candidates(nodes)
        if extract_only:
            print("\nDone! Review the CSV, then re-run without --extract.")
            return

    # Load reviewed words
    wc_words = load_wordcloud_words()
    print(f"  Loaded {len(wc_words)} reviewed words for word cloud")

    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    print("Plotting (a) modality comparison...")
    plot_modality_comparison(ax_a)

    print("Plotting (b) node type distribution...")
    plot_node_type_distribution(ax_b, nodes)

    print("Plotting (c) CPMP heatmap...")
    plot_cpmp_heatmap(ax_c, cpmp_results, cpmp_cats)

    print("Plotting (d) community distribution...")
    plot_community_distribution(ax_d, nodes)

    print("Plotting (e) Neo4j placeholder...")
    plot_neo4j_placeholder(ax_e)

    print("Plotting (f) word cloud...")
    plot_wordcloud(ax_f, wc_words)

    # Save
    for fmt in ["png", "svg", "pdf"]:
        out = OUTPUT_DIR / f"fig2_kg_statistics.{fmt}"
        fig.savefig(out, format=fmt, dpi=300,
                    bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
