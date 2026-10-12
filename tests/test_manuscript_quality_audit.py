from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from manuscript_quality_audit import audit_manuscript  # noqa: E402


def _write_fixture(root: Path, prose: str) -> None:
    paper = root / "paper"
    generated = paper / "generated"
    generated.mkdir(parents=True)
    (generated / "results_macros.tex").write_text(
        "\\newcommand{\\NIncludedInstances}{4}\n", encoding="utf-8"
    )
    (paper / "main.tex").write_text(prose, encoding="utf-8")


def test_audit_warns_without_rewriting_manuscript(tmp_path):
    manuscript = r"""\begin{abstract}
Production planning under uncertainty requires linked timing and resource decisions. However, it remains unclear when expensive refinement is justified. We compare a model and three methods. Four complete benchmark instances produced numeric results: runtime fell by 75 percent. The evidence supports contextual allocation of computational effort.
\end{abstract}
\section{Introduction}
The scanner reads metadata.json from results/example.
\citep{Known,Missing}
\begin{figure}\label{fig:used}\end{figure}
Figure~\ref{fig:used} reports the result.
\begin{thebibliography}{}\bibitem{Known}Known.\end{thebibliography}
"""
    _write_fixture(tmp_path, manuscript)
    before = (tmp_path / "paper" / "main.tex").read_text(encoding="utf-8")
    path, status, issues = audit_manuscript(tmp_path)
    after = (tmp_path / "paper" / "main.tex").read_text(encoding="utf-8")
    report = path.read_text(encoding="utf-8")
    assert before == after
    assert status == "NOT READY — ISSUES DETECTED"
    assert any("Repository-specific" in issue for issue in issues)
    assert "Missing" in report


def test_audit_can_reach_manual_review_state(tmp_path):
    abstract = (
        "Production planning becomes difficult when discrete timing, continuous resource allocation, and uncertain economic conditions interact. "
        "Agricultural systems make this coupling explicit through land occupation and correlated selling prices. "
        "However, it remains unclear when mathematical refinement should be exhaustive and when selective allocation is sufficient. "
        "We compare an exact reference, evolutionary search, full refinement, and surrogate-guided refinement in a replicated benchmark design. "
        "Across four complete benchmark instances, selective refinement reduced computational effort by 75 percent while preserving useful trade-offs. "
        "The evidence supports allocating expensive optimization effort according to planning scale and acceptable quality loss. "
        "This balance connects certification, Pareto quality, and response time in modern production planning. "
        "The resulting framework remains interpretable to decision makers while retaining explicit economic-risk objectives. "
        "Selective allocation is valuable when refinement dominates runtime, but its savings must be interpreted jointly with any observed loss of Pareto quality. "
        "Matched seeds and common initialization isolate that allocation choice from differences in representation, operators, or starting populations. "
        "Pooled empirical references also prevent a time-limited exact archive from receiving automatically favourable distance indicators. "
        "Together, these controls make the comparison relevant to planners who must balance evidence quality with response time. "
        "Its principal implication is that hybrid intensity should be treated as a planning-resource decision rather than a fixed algorithmic choice."
    )
    manuscript = rf"""\begin{{abstract}}{abstract}\end{{abstract}}
\section{{Introduction}}Scientific prose.\n"""
    _write_fixture(tmp_path, manuscript)
    _, status, issues = audit_manuscript(tmp_path)
    assert not issues
    assert status == "READY FOR MANUAL SCIENTIFIC REVIEW"
