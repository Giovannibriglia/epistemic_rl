from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from src.utils import preprocess_sample

COL_NAMES_CSV = [
    "Path Hash",
    "Path Hash Merged",
    "Path Mapped",
    "Path Mapped Merged",
    "Depth",
    "Distance From Goal",
    "Goal",
]


class GraphDataPipeline:

    def __init__(
        self,
        folder_data: str,
        kind_of_ordering: str,
        kind_of_data: str,
        max_samples_for_prob: int,
        max_unreachable_ratio: float = 0.3,
        use_goal: bool = True,
        use_depth: bool = True,
        UNREACHABLE_STATE_VALUE: int = 1_000_000,
        GOAL_DISTANCE_TO_REPLACE: int | None = -1,
    ):
        self.folder_data = Path(folder_data)
        self.ordering = kind_of_ordering
        self.data_kind = kind_of_data
        self.max_samples = max_samples_for_prob
        self.max_unreachable_ratio = max_unreachable_ratio
        self.use_goal = use_goal
        self.use_depth = use_depth
        self.UNREACHABLE_STATE_VALUE = UNREACHABLE_STATE_VALUE
        self.GOAL_DISTANCE_TO_REPLACE = GOAL_DISTANCE_TO_REPLACE

        self.df: Optional[pd.DataFrame] = None
        self.samples: List[Dict[str, Any]] = []

        self._build_df()
        self._load_samples()

    def _get_all_items(self, folder: Path) -> List[Path]:
        return list(folder.iterdir())

    def _read_csv(self, csv_path: Path) -> pd.DataFrame:
        records = []
        with csv_path.open(newline="") as f:
            next(f)  # skip header
            for raw in f:
                parts = raw.rstrip("\n").split(",", len(COL_NAMES_CSV) - 1)
                records.append(parts)
        df = pd.DataFrame(records, columns=COL_NAMES_CSV)
        df["Depth"] = pd.to_numeric(df["Depth"], errors="coerce")
        df["Distance From Goal"] = pd.to_numeric(
            df["Distance From Goal"], errors="coerce"
        )

        # determine which path column to keep
        key = f"Path {'Hash' if self.ordering == 'hash' else 'Mapped'}"
        key += " Merged" if self.data_kind == "merged" else ""
        df = df[["Depth", "Distance From Goal", key, "Goal"]]
        df = df.rename(columns={key: "Path State"})
        return df

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df["Distance From Goal"].value_counts()
        maj = self.UNREACHABLE_STATE_VALUE
        maj_count = counts.get(maj, 0)
        max_maj = int(self.max_samples * self.max_unreachable_ratio)
        maj_df = df[df["Distance From Goal"] == maj].sample(
            n=min(max_maj, maj_count), random_state=0
        )

        other = counts.drop(maj, errors="ignore")
        remaining = self.max_samples - len(maj_df)
        per_cls = remaining // max(1, len(other))

        others = []
        for cls, cnt in other.items():
            take = min(per_cls, cnt)
            sampled = df[df["Distance From Goal"] == cls].sample(n=take, random_state=0)
            others.append(sampled)

        return pd.concat([maj_df, *others], ignore_index=True).sample(
            frac=1, random_state=0
        )

    def _build_df(self) -> pd.DataFrame:
        frames = []
        for prob_dir in self._get_all_items(self.folder_data):
            csv = next(
                (p for p in self._get_all_items(prob_dir) if p.suffix == ".csv"), None
            )
            if not csv:
                continue
            df = self._read_csv(csv)
            print(df.value_counts(df["Distance From Goal"]))

            df["Distance From Goal"] = df["Distance From Goal"].replace(
                self.GOAL_DISTANCE_TO_REPLACE,
                self.UNREACHABLE_STATE_VALUE,
            )
            df = self._balance_dataset(df)
            frames.append(df)
        self.df = pd.concat(frames, ignore_index=True)

        print(self.df.value_counts(self.df["Distance From Goal"]))

        return self.df

    def _load_samples(self) -> List[Dict[str, Any]]:
        if self.df is None:
            raise ValueError("Call build_df() first.")
        for _, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Building samples..."
        ):
            s = preprocess_sample(
                row["Path State"],
                int(row["Depth"]) if self.use_depth else None,
                int(row["Distance From Goal"]),
                row["Goal"] if self.use_goal else None,
            )
            self.samples.append(s)
        return self.samples

    def save(self, out_dir: str, extra_params: Optional[Dict[str, Any]] = None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {
                "folder_data": str(self.folder_data),
                "ordering": self.ordering,
                "data_kind": self.data_kind,
                "max_samples": self.max_samples,
                "use_goal": self.use_goal,
                **(extra_params or {}),
            },
            "df": self.df,
            "samples": self.samples,
        }
        torch.save(payload, out / "dataloader_info.pt")
        return out / "dataloader_info.pt"

    def get_df(self):
        return self.df

    def get_samples(self):
        return self.samples
