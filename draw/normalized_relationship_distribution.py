import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SPECIAL_CASES = {
    "HAS_NO_EFFECT_ON": "NO_EFFECT_OR_EXCLUDES",
    "DOES_NOT_AFFECT": "NO_EFFECT_OR_EXCLUDES",
    "EXCLUDES": "NO_EFFECT_OR_EXCLUDES",
    "LACKS": "NO_EFFECT_OR_EXCLUDES",
    "OUTPERFORMS": "COMPARES_WITH",
    "IMPROVES_UPON": "COMPARES_WITH",
    "VARIES_WITH": "CORRELATES_WITH",
    "PARAMETER_OF": "HAS_OR_EXHIBITS",
    "COMPONENT_OF": "CONTAINS_OR_PART_OF",
    "MADE_OF": "CONTAINS_OR_PART_OF",
    "COMPRISES": "CONTAINS_OR_PART_OF",
    "USED_AS": "USES_OR_APPLIES_TO",
    "APPLIED_IN": "USES_OR_APPLIES_TO",
    "EMPLOYS": "USES_OR_APPLIES_TO",
    "EXAMINES": "OBSERVES_OR_MEASURES",
    "DEMONSTRATES": "OBSERVES_OR_MEASURES",
    "REPORTS": "OBSERVES_OR_MEASURES",
    "VERIFIES": "OBSERVES_OR_MEASURES",
    "CAPTURES": "OBSERVES_OR_MEASURES",
    "MODELED_BY": "MODELS_OR_PREDICTS",
    "COMPUTES": "MODELS_OR_PREDICTS",
    "ESTIMATES": "MODELS_OR_PREDICTS",
    "MANUFACTURED_BY": "PROCESSED_OR_SUBJECTED_TO",
    "TREATED_WITH": "PROCESSED_OR_SUBJECTED_TO",
    "COATED_WITH": "PROCESSED_OR_SUBJECTED_TO",
    "DOPES": "PROCESSED_OR_SUBJECTED_TO",
    "CAUSES_FORMATION_OF": "CAUSES_OR_CONTRIBUTES_TO",
    "STRENGTHENED_BY": "PROMOTES_OR_INCREASES",
    "WEAKENS": "REDUCES_OR_INHIBITS",
    "DEPOSITED_ON": "LOCATED_OR_OCCURS_IN",
    "OCCURS_DURING": "TEMPORAL_ORDER",
    "DEVELOPED": "PRODUCES_OR_TRANSFORMS_TO",
    "DEVELOPS": "PRODUCES_OR_TRANSFORMS_TO",
    "PRECIPITATES": "PRODUCES_OR_TRANSFORMS_TO",
    "EVOLVES_INTO": "PRODUCES_OR_TRANSFORMS_TO",
    "DRIVES": "CAUSES_OR_CONTRIBUTES_TO",
    "PROVIDES": "PROMOTES_OR_INCREASES",
    "ACHIEVES": "PRODUCES_OR_TRANSFORMS_TO",
    "ATTRIBUTED_TO": "CAUSES_OR_CONTRIBUTES_TO",
    "INFORMS": "DESCRIBES_OR_DEFINES",
    "PROCESSES": "PROCESSED_OR_SUBJECTED_TO",
    "FABRICATES": "PROCESSED_OR_SUBJECTED_TO",
    "INCORPORATES": "CONTAINS_OR_PART_OF",
    "INVOLVES": "CONTAINS_OR_PART_OF",
    "DERIVED_FROM": "PROCESSED_OR_SUBJECTED_TO",
    "ORIGINATES_FROM": "PROCESSED_OR_SUBJECTED_TO",
    "DISSOLVES": "PRODUCES_OR_TRANSFORMS_TO",
    "ACTIVATES": "PROMOTES_OR_INCREASES",
    "EXTENDS": "CONNECTS_OR_FLOWS_TO",
    "JOINS": "CONNECTS_OR_FLOWS_TO",
    "INTERFACES_WITH": "CONNECTS_OR_FLOWS_TO",
    "RESISTS": "REDUCES_OR_INHIBITS",
    "CONSTRAINS": "DEPENDS_ON_OR_REQUIRES",
    "GOVERNED_BY": "DEPENDS_ON_OR_REQUIRES",
    "AGREES_WITH": "COMPARES_WITH",
    "CONSISTENT_WITH": "COMPARES_WITH",
    "SIMILAR_TO": "COMPARES_WITH",
    "RESEMBLES": "COMPARES_WITH",
    "ALIGNS_WITH": "COMPARES_WITH",
    "ALIGNED_WITH": "COMPARES_WITH",
    "HAS_COUNTERPART": "COMPARES_WITH",
    "IS_A_TYPE_OF": "IS_A",
    "CLASSIFIED_AS": "IS_A",
    "AFFILIATED_WITH": "PUBLICATION_METADATA",
    "EXCEEDS": "COMPARES_WITH",
    "PROTECTS": "PROMOTES_OR_INCREASES",
    "PRESERVES": "PROMOTES_OR_INCREASES",
    "PARTICIPATES_IN": "PROCESSED_OR_SUBJECTED_TO",
    "EQUIPPED_WITH": "CONTAINS_OR_PART_OF",
    "DEPOSITS": "PRODUCES_OR_TRANSFORMS_TO",
    "GROUPED_WITH": "COMPARES_WITH",
    "DOMINATES": "AFFECTS_OR_INFLUENCES",
    "EXEMPLIFIES": "DESCRIBES_OR_DEFINES",
    "HAS_STRUCTURE": "HAS_OR_EXHIBITS",
    "FEATURES": "HAS_OR_EXHIBITS",
    "SUBJECT_TO": "PROCESSED_OR_SUBJECTED_TO",
    "GUIDES": "DESCRIBES_OR_DEFINES",
    "SUITABLE_FOR": "USES_OR_APPLIES_TO",
}


PATTERN_GROUPS = [
    ("PUBLICATION_METADATA", ("AUTHOR", "PUBLISH", "FUND", "AFFILIAT", "ACKNOWLEDG", "REFERENCE")),
    ("IS_A", ("IS_A", "IS_TYPE", "INSTANCE", "CLASSIFIED_AS", "VARIANT_OF")),
    (
        "CONTAINS_OR_PART_OF",
        ("CONTAIN", "INCLUDE", "CONSIST", "COMPOSE", "COMPRIS", "PART", "COMPONENT", "ELEMENT", "BELONG", "INCORPORAT", "INVOLVE", "EQUIPPED"),
    ),
    ("HAS_OR_EXHIBITS", ("HAS_", "HAS", "EXHIBIT", "DISPLAY", "MANIFEST", "FEATURE")),
    ("AFFECTS_OR_INFLUENCES", ("AFFECT", "INFLUENC", "IMPACT", "MODULAT", "ALTER", "MODIF", "DETERMINE", "CONTROL", "GOVERN", "DOMINAT")),
    ("CAUSES_OR_CONTRIBUTES_TO", ("CAUSE", "INDUC", "TRIGGER", "LEAD", "RESULT", "CONTRIBUT", "DRIVE", "ATTRIBUTED")),
    ("PROMOTES_OR_INCREASES", ("INCREASE", "ENHANC", "IMPROV", "PROMOT", "ENABLE", "FACILIT", "STRENGTHEN", "ACCELERAT", "STABILIZ", "SUPPORT", "ACTIVAT", "PROVID", "PROTECT", "PRESERV")),
    ("REDUCES_OR_INHIBITS", ("REDUC", "DECREAS", "SUPPRESS", "INHIBIT", "HINDER", "LIMIT", "PREVENT", "ELIMINAT", "DEGRAD", "MITIGAT", "WEAKEN", "MINIMIZ", "RESIST", "INTERFERE")),
    ("CORRELATES_WITH", ("CORRELAT", "ASSOCIAT", "RELAT", "CORRESPOND", "MATCH", "VARY", "EQUAL")),
    ("USES_OR_APPLIES_TO", ("USES", "USED", "UTILIZ", "APPLI", "EMPLOY", "SUITABLE_FOR")),
    ("OBSERVES_OR_MEASURES", ("OBSERV", "ANALYZ", "MEASUR", "CHARACTERIZ", "DETECT", "IDENTIF", "REVEAL", "SHOW", "INDICAT", "CONFIRM", "EVALUAT", "TEST", "STUDI", "INVESTIGAT", "EXAMIN", "DEMONSTRAT", "VERIFY", "CAPTUR", "REPORT")),
    ("DESCRIBES_OR_DEFINES", ("DESCRIB", "DEFIN", "REPRESENT", "EXPLAIN", "ILLUSTRAT", "EXEMPLIF", "INFORM", "GUIDE")),
    ("MODELS_OR_PREDICTS", ("MODEL", "PREDICT", "SIMULAT", "CALCULAT", "QUANTIF", "VALIDAT", "ESTIMAT", "COMPUT")),
    ("PRODUCES_OR_TRANSFORMS_TO", ("PRODUC", "GENERAT", "FORM", "TRANSFORM", "TRANSITION", "YIELD", "CREATE", "PRECIPITAT", "DEVELOP", "DISSOLV", "DEPOSIT", "EVOLVE", "INTRODUC", "REPLACE", "SEPARAT")),
    ("PROCESSED_OR_SUBJECTED_TO", ("PROCESS", "PREPAR", "FABRICAT", "MANUFACTUR", "ADD", "SUBJECT", "UNDERGO", "PERFORM", "EXPOS", "DERIV", "ORIGINAT", "PARTICIPAT", "TREAT", "COAT", "DOPE", "IMPLEMENT")),
    ("LOCATED_OR_OCCURS_IN", ("ADJACENT", "DISTRIBUT", "LOCAT", "SURROUND", "OCCUR", "PRESENT", "SEGREGAT")),
    ("CONNECTS_OR_FLOWS_TO", ("FLOW", "CONNECT", "JOIN", "EXTEND", "INTERFACE")),
    ("COMPARES_WITH", ("DIFFER", "COMPARE", "CONTRAST", "ALTERNATIVE", "SIMILAR", "AGREE", "CONSISTENT", "RESEMBL", "MATCH", "COUNTERPART", "OUTPERFORM", "EXCEED", "GROUPED", "ALIGN")),
    ("INTERACTS_WITH", ("INTERACT", "COMBIN", "REACT", "COEXIST", "COLLABORAT")),
    ("DEPENDS_ON_OR_REQUIRES", ("DEPEND", "REQUIR", "BASED", "CONSTRAIN")),
    ("TEMPORAL_ORDER", ("FOLLOW", "PRECEDE", "DURING")),
    ("NO_EFFECT_OR_EXCLUDES", ("NO_EFFECT", "DOES_NOT", "EXCLUD", "LACK")),
]


def iter_relation_types(graph_path: Path):
    for key in ("links", "relationships"):
        try:
            yield from _iter_array_items(graph_path, key)
            return
        except RuntimeError:
            continue
    raise RuntimeError('Neither "links" nor "relationships" array was found in the graph JSON.')


def _iter_array_items(graph_path: Path, array_key: str):
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024

    with graph_path.open("r", encoding="utf-8") as handle:
        buffer = ""
        in_array = False
        eof = False

        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            if not in_array:
                key_pos = buffer.find(f'"{array_key}"')
                if key_pos == -1:
                    if eof:
                        raise RuntimeError(f'Could not find "{array_key}" array in JSON.')
                    buffer = buffer[-64:]
                    continue

                bracket_pos = buffer.find("[", key_pos)
                if bracket_pos == -1:
                    if eof:
                        raise RuntimeError(f'Could not find start of "{array_key}" array in JSON.')
                    continue

                buffer = buffer[bracket_pos + 1 :]
                in_array = True

            progressed = False
            while True:
                stripped = buffer.lstrip()
                if stripped != buffer:
                    buffer = stripped
                    progressed = True

                if not buffer:
                    break

                if buffer[0] == "]":
                    return

                if buffer[0] == ",":
                    buffer = buffer[1:]
                    progressed = True
                    continue

                try:
                    obj, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break

                yield obj
                buffer = buffer[end:]
                progressed = True

            if eof and not progressed:
                raise RuntimeError(f'Unexpected EOF while parsing "{array_key}" array.')


def normalize_relation_type(relation_type: str) -> str:
    raw = (relation_type or "UNKNOWN").upper().strip()
    if raw in SPECIAL_CASES:
        return SPECIAL_CASES[raw]

    for normalized, patterns in PATTERN_GROUPS:
        if any(pattern in raw for pattern in patterns):
            return normalized

    return "OTHER"


def analyze(graph_path: Path):
    raw_counter = Counter()
    normalized_counter = Counter()
    normalized_sources = defaultdict(Counter)

    for item in iter_relation_types(graph_path):
        raw_relation = item.get("relationship") or item.get("label") or item.get("relation") or "UNKNOWN"
        normalized_relation = normalize_relation_type(raw_relation)
        raw_counter[raw_relation] += 1
        normalized_counter[normalized_relation] += 1
        normalized_sources[normalized_relation][raw_relation] += 1

    return raw_counter, normalized_counter, normalized_sources


def write_distribution_csv(output_path: Path, normalized_counter: Counter, normalized_sources, total_edges: int):
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["normalized_relation", "count", "proportion", "raw_type_count", "top_raw_examples"])
        for relation, count in normalized_counter.most_common():
            examples = "; ".join(f"{raw} ({raw_count})" for raw, raw_count in normalized_sources[relation].most_common(5))
            writer.writerow(
                [
                    relation,
                    count,
                    f"{count / total_edges:.4%}",
                    len(normalized_sources[relation]),
                    examples,
                ]
            )


def write_mapping_csv(output_path: Path, raw_counter: Counter):
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raw_relation", "normalized_relation", "count"])
        for raw_relation, count in raw_counter.most_common():
            writer.writerow([raw_relation, normalize_relation_type(raw_relation), count])


def write_summary_json(output_path: Path, raw_counter: Counter, normalized_counter: Counter, normalized_sources):
    total_edges = sum(raw_counter.values())
    other_edges = normalized_counter.get("OTHER", 0)
    payload = {
        "total_edges": total_edges,
        "unique_raw_relation_types": len(raw_counter),
        "unique_normalized_relation_types": len(normalized_counter),
        "normalized_coverage_edges": total_edges - other_edges,
        "normalized_coverage_ratio": round((total_edges - other_edges) / total_edges, 6) if total_edges else 0.0,
        "other_edges": other_edges,
        "other_ratio": round(other_edges / total_edges, 6) if total_edges else 0.0,
        "top_normalized_relations": [
            {
                "normalized_relation": relation,
                "count": count,
                "proportion": round(count / total_edges, 6) if total_edges else 0.0,
                "raw_type_count": len(normalized_sources[relation]),
                "top_raw_examples": [
                    {"raw_relation": raw, "count": raw_count}
                    for raw, raw_count in normalized_sources[relation].most_common(5)
                ],
            }
            for relation, count in normalized_counter.most_common(25)
        ],
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def plot_distribution(output_path: Path, normalized_counter: Counter, total_edges: int, top_n: int, include_other: bool):
    items = normalized_counter.most_common()
    if not include_other:
        items = [(relation, count) for relation, count in items if relation != "OTHER"]
    items = items[:top_n]

    if not items:
        raise RuntimeError("No normalized relation types were available for plotting.")

    labels = [relation for relation, _ in items]
    counts = [count for _, count in items]
    proportions = [count / total_edges for count in counts]

    plt.rcParams.update({"font.size": 12, "font.family": "Arial"})
    fig_height = max(6, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    colors = plt.cm.Blues_r([0.15 + 0.75 * i / max(1, len(labels) - 1) for i in range(len(labels))])

    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Edge Count")
    title_suffix = "including OTHER" if include_other else "excluding OTHER"
    ax.set_title(f"Normalized Relationship Type Distribution ({title_suffix})")
    ax.grid(axis="x", alpha=0.25)

    max_count = max(counts)
    for idx, (bar, count, proportion) in enumerate(zip(bars, counts, proportions)):
        ax.text(
            count + max_count * 0.01,
            idx,
            f"{count:,} ({proportion:.1%})",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".svg"), format="svg", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize relation types in final_graph.json and draw their distribution.")
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "graphs" / "final_graph.json",
        help="Path to final_graph.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations",
        help="Directory for CSV / JSON / figure outputs",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of normalized relation types to draw",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_counter, normalized_counter, normalized_sources = analyze(args.graph_path)
    total_edges = sum(raw_counter.values())

    write_distribution_csv(
        output_dir / "normalized_relationship_type_distribution.csv",
        normalized_counter,
        normalized_sources,
        total_edges,
    )
    write_mapping_csv(output_dir / "normalized_relationship_type_mapping.csv", raw_counter)
    write_summary_json(
        output_dir / "normalized_relationship_type_summary.json",
        raw_counter,
        normalized_counter,
        normalized_sources,
    )
    plot_distribution(
        output_dir / "normalized_relationship_type_distribution",
        normalized_counter,
        total_edges,
        args.top_n,
        include_other=True,
    )
    plot_distribution(
        output_dir / "normalized_relationship_type_distribution_no_other",
        normalized_counter,
        total_edges,
        args.top_n,
        include_other=False,
    )

    print(f"Total edges: {total_edges}")
    print(f"Unique raw relation types: {len(raw_counter)}")
    print(f"Unique normalized relation types: {len(normalized_counter)}")
    print(f"Coverage without OTHER: {(total_edges - normalized_counter.get('OTHER', 0)) / total_edges:.2%}")
    print("Top normalized relation types:")
    for relation, count in normalized_counter.most_common(15):
        print(f"  {relation}: {count} ({count / total_edges:.2%})")


if __name__ == "__main__":
    main()
