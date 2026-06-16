from dataclasses import dataclass
from typing import Iterable, List, Optional

import csv


@dataclass(frozen=True)
class ReviewAction:
    sample_id: str
    action: str
    reason: str


def load_review_actions(csv_path: Optional[str]) -> List[ReviewAction]:
    """Load review actions from a CSV file.

    Expected columns: sound_id, action, reason
    """
    if not csv_path:
        return []

    actions: List[ReviewAction] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actions.append(
                ReviewAction(
                    sample_id=str(row.get("sound_id", row.get("sample_id", ""))).strip(),
                    action=str(row.get("action", "")).strip(),
                    reason=str(row.get("reason", "")).strip(),
                )
            )
    return actions


def filter_sample_ids(sample_ids: Iterable[str], actions: List[ReviewAction]) -> List[str]:
    """Filter sample ids based on review actions.

    Actions are interpreted by downstream policy. This helper only returns
    ids that are not marked for exclusion (action == 'exclude').
    """
    excluded = {a.sample_id for a in actions if a.action == "exclude"}
    return [sid for sid in sample_ids if sid not in excluded]
