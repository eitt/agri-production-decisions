"""Warn about narrative-integrity regressions in the scientific manuscript.

The audit is deliberately conservative: it reports issues and candidates for
manual scientific review, but it never rewrites claims or prose.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}
REPOSITORY_PATTERNS = [
    r"\b[\w.-]+\.(?:csv|json|py|ya?ml)\b",
    r"\b(?:results|data|paper/generated)/[^\s,;)}]+",
    r"\b(?:run[_ -]?id|snapshot scanner|GitHub path|repository path|"
    r"local folder|source file|generated artifact|pipeline output)\b",
]
STYLE_TERMS = [
    "current snapshot", "publication-safe", "publication-complete", "pipeline",
    "generated artifact", "configured benchmark", "submission snapshot",
]
PLACEHOLDERS = [r"\bTODO\b", r"\bTBD\b", r"\bXXX\b", r"\?\?", r"<[^>]+>"]


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _prose_lines(text: str) -> list[tuple[int, str]]:
    lines = []
    for number, raw in enumerate(_strip_comments(text).splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("\\input", "\\includegraphics", "\\newcommand")):
            continue
        lines.append((number, stripped))
    return lines


def _environment(text: str, name: str) -> str:
    match = re.search(rf"\\begin\{{{name}\}}(.*?)\\end\{{{name}\}}", text, re.S)
    return match.group(1).strip() if match else ""


def _plain_latex(text: str) -> str:
    text = re.sub(r"\\(?:cite[tp]?|ref|label)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", _plain_latex(text), flags=re.UNICODE))


def _macro_int(generated: Path, name: str) -> int | None:
    if not generated.exists():
        return None
    match = re.search(rf"\\newcommand\{{\\{re.escape(name)}\}}\{{(\d+)\}}", generated.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _labels_from_includes(main_text: str, paper: Path) -> set[str]:
    labels = set(re.findall(r"\\label\{([^}]+)\}", main_text))
    for relative in re.findall(r"\\input\{([^}]+)\}", main_text):
        candidate = paper / relative
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".tex")
        if candidate.exists():
            labels.update(re.findall(r"\\label\{([^}]+)\}", candidate.read_text(encoding="utf-8")))
    return labels


def _subsection_counts(text: str) -> list[tuple[str, int]]:
    sections = list(re.finditer(r"^\\section\{([^}]+)\}", text, re.M))
    output = []
    for index, match in enumerate(sections):
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        count = len(re.findall(r"^\\subsection\{", text[match.end():end], re.M))
        output.append((match.group(1), count))
    return output


def _instance_claim_mismatches(text: str, expected: int | None) -> list[str]:
    if expected is None:
        return ["Generated macro NIncludedInstances could not be read."]
    candidates = []
    pattern = re.compile(
        r"\b(?P<n>\d+|" + "|".join(NUMBER_WORDS) + r")\s+"
        r"(?:(?:publication[- ]complete|complete|included)\s+)?"
        r"(?:data-informed synthetic\s+)?benchmark instances\b",
        re.I,
    )
    for match in pattern.finditer(_plain_latex(text)):
        token = match.group("n").lower()
        value = int(token) if token.isdigit() else NUMBER_WORDS[token]
        phrase = match.group(0)
        if value != expected and any(term in phrase.lower() for term in ("complete", "included")):
            candidates.append(f"'{phrase}' conflicts with generated count {expected}.")
    return candidates


def _latex_diagnostics(paper: Path) -> tuple[str, str]:
    log_path = paper / "main.log"
    if not log_path.exists():
        return "unavailable", "unavailable"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    page_match = re.search(r"Output written on .*?\((\d+) pages?", log)
    pages = page_match.group(1) if page_match else "unavailable"
    warnings = len(re.findall(r"(?:LaTeX(?: Font)?|Package|Class|pdfTeX) Warning:", log))
    return pages, str(warnings)


def audit_manuscript(root: Path = ROOT) -> tuple[Path, str, list[str]]:
    paper = root / "paper"
    main_path = paper / "main.tex"
    output_path = paper / "generated" / "manuscript_quality_audit.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = main_path.read_text(encoding="utf-8")
    prose = _prose_lines(text)
    issues: list[str] = []

    repository_hits = []
    for line_number, line in prose:
        for pattern in REPOSITORY_PATTERNS:
            for match in re.finditer(pattern, line, re.I):
                repository_hits.append(f"line {line_number}: {match.group(0)}")
    if repository_hits:
        issues.append(f"Repository-specific language remains in manuscript prose ({len(repository_hits)} hit(s)).")

    style_hits = []
    for line_number, line in prose:
        for term in STYLE_TERMS:
            if term.lower() in line.lower():
                style_hits.append(f"line {line_number}: {term}")

    placeholder_hits = []
    for line_number, line in prose:
        for pattern in PLACEHOLDERS:
            if re.search(pattern, line):
                placeholder_hits.append(f"line {line_number}: {line}")
                break
    if placeholder_hits:
        issues.append(f"Unresolved placeholders detected ({len(placeholder_hits)} hit(s)).")

    abstract = _environment(text, "abstract")
    abstract_words = _word_count(abstract)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", _plain_latex(abstract)) if part.strip()]
    opening = " ".join(sentences[:2]).lower()
    first_three = " ".join(sentences[:3]).lower()
    final_sentence = sentences[-1].lower() if sentences else ""
    abstract_checks = {
        "problem-first opening": bool(re.search(r"planning|production|decision|uncertain|risk", opening)),
        "explicit unresolved gap before methods": bool(re.search(r"unclear|unresolved|remains|gap|however|yet", first_three)),
        "methodology after gap": bool(re.search(r"model|experiment|compare|method|epsilon|nsga", _plain_latex(abstract).lower())),
        "numeric decision-relevant results": bool(re.search(r"\d", abstract)),
        "interpretive final sentence": bool(re.search(r"evidence|result|support|indicat|show|impli|trade-off", final_sentence)),
        "target length 180--230 words": 180 <= abstract_words <= 230,
    }
    failed_abstract = [name for name, passed in abstract_checks.items() if not passed]
    if failed_abstract:
        issues.append("Abstract structural checklist failed: " + ", ".join(failed_abstract) + ".")

    expected_instances = _macro_int(paper / "generated" / "results_macros.tex", "NIncludedInstances")
    stale_claims = _instance_claim_mismatches(text, expected_instances)
    if stale_claims:
        issues.append(f"Potential stale instance-count claims detected ({len(stale_claims)}).")

    labels = _labels_from_includes(text, paper)
    references = set(re.findall(r"\\ref\{([^}]+)\}", text))
    missing_reference_targets = sorted(references - labels)
    unreferenced_display_labels = sorted(label for label in labels if label.startswith(("fig:", "tab:")) and label not in references)
    if missing_reference_targets:
        issues.append(f"Missing figure/table reference targets detected ({len(missing_reference_targets)}).")
    if unreferenced_display_labels:
        issues.append(f"Unreferenced figure/table labels detected ({len(unreferenced_display_labels)}).")

    cited = set()
    for group in re.findall(r"\\cite(?:p|t)?\{([^}]+)\}", text):
        cited.update(key.strip() for key in group.split(",") if key.strip())
    bibliography = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", text))
    unresolved_citations = sorted(cited - bibliography)
    uncited_bibliography = sorted(bibliography - cited)
    if unresolved_citations:
        issues.append(f"Unresolved citation keys detected ({len(unresolved_citations)}).")

    repeated_numbers = Counter(re.findall(r"\b\d+(?:[.,]\d+)?\b", _plain_latex(text)))
    repeated_claims = sorted((value, count) for value, count in repeated_numbers.items() if count >= 4)
    pages, latex_warnings = _latex_diagnostics(paper)
    status = "READY FOR MANUAL SCIENTIFIC REVIEW" if not issues else "NOT READY — ISSUES DETECTED"

    lines = [
        "MANUSCRIPT NARRATIVE QUALITY AUDIT",
        "=" * 38,
        f"Manuscript: {main_path}",
        f"Expected complete instances: {expected_instances if expected_instances is not None else 'unavailable'}",
        f"Abstract word count: {abstract_words}",
        "",
        "ABSTRACT STRUCTURAL CHECKLIST",
        *[f"- {'PASS' if passed else 'WARN'}: {name}" for name, passed in abstract_checks.items()],
        "",
        "SUBSECTIONS PER SECTION",
        *[f"- {section}: {count}" for section, count in _subsection_counts(text)],
        "",
        "REPOSITORY-SPECIFIC LANGUAGE IN PROSE",
        *(repository_hits or ["- None detected."]),
        "",
        "STYLE TERMS FOR MANUAL REVIEW",
        *(style_hits or ["- None detected."]),
        "",
        "UNRESOLVED PLACEHOLDERS",
        *(placeholder_hits or ["- None detected."]),
        "",
        "STALE INSTANCE-COUNT CLAIMS",
        *(stale_claims or ["- None detected."]),
        "",
        "FIGURE/TABLE REFERENCES",
        f"- Missing targets: {', '.join(missing_reference_targets) if missing_reference_targets else 'none'}",
        f"- Unreferenced labels: {', '.join(unreferenced_display_labels) if unreferenced_display_labels else 'none'}",
        "",
        "CITATIONS",
        f"- Unresolved keys: {', '.join(unresolved_citations) if unresolved_citations else 'none'}",
        f"- Uncited bibliography entries: {', '.join(uncited_bibliography) if uncited_bibliography else 'none'}",
        "",
        "REPEATED NUMERICAL CLAIM CANDIDATES",
        *( [f"- {value}: {count} occurrences" for value, count in repeated_claims]
           or ["- No number occurs four or more times."] ),
        "- These are candidates for manual comparison with generated artifacts; repetition alone is not an error.",
        "",
        "LATEX BUILD SIGNALS",
        f"- Final page count from main.log: {pages}",
        f"- LaTeX/package/class warnings in main.log: {latex_warnings}",
        "",
        "ISSUES",
        *([f"- {issue}" for issue in issues] or ["- None detected by automated checks."]),
        "",
        status,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path, status, issues


def main() -> None:
    path, status, _ = audit_manuscript()
    print(path)
    print(status)


if __name__ == "__main__":
    main()
