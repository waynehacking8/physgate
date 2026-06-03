"""Evaluation methodology v2 (docs/design/EVALUATION_METHODOLOGY.md).

Research-grounded experiments replacing the ad-hoc v1 orchestration suite as
the project's primary evidence:

    E1  pipeline ablation        — value of each validation layer end to end
    E2  gate-as-classifier       — precision/recall/F1 over a defect corpus
    E3  procedural scenarios     — generated layouts with certified solvability
    E4  statistical repeats      — dispersion of all stochastic measurements
"""

__all__: list[str] = []
