from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.data import canonical_submission_path, copy_file
from models.final_meta_regime_ensemble import build_final_meta_regime_ensemble


def main() -> None:
    meta_outputs = build_final_meta_regime_ensemble()
    canonical_path = copy_file(meta_outputs["final_submission"], canonical_submission_path())
    print("Final submission model:", meta_outputs["final_candidate_name"])
    print("Internal CV winner:", meta_outputs["selected_config"]["internal_cv_winner_name"])
    print("Final submission objective:", float(meta_outputs["practical_objective"]))
    print("Final submission file:", meta_outputs["final_submission"])
    print("Canonical submission:", canonical_path)


if __name__ == "__main__":
    main()
