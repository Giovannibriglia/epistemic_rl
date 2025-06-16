import os
import random
import re
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, SAGEConv
from torch_geometric.utils import from_networkx
from tqdm import tqdm

COL_NAMES_CSV = [
    "Path Hash",
    "Path Hash Merged",
    "Path Mapped",
    "Path Mapped Merged",
    "Depth",
    "Distance From Goal",
    "Goal",
]

UNREACHABLE_STATE_VALUE = -99999
# You asked for this explicitly 🙂
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


def _get_all_items(folder_path):
    folder = Path(folder_path)
    return [item for item in folder.iterdir()]


def _read_csv(csv_file_path: str, kind_of_ordering: str, kind_of_data: str):
    records = []

    with open(csv_file_path, newline="") as f:
        next(f)  # skip original header
        for raw in f:
            # Keep everything after the 3rd comma together as "Goal"
            path_hash, path_hash_m, path_map, path_map_m, depth, dist, goal = (
                raw.rstrip("\n").split(",", len(COL_NAMES_CSV) - 1)
            )
            records.append(
                [path_hash, path_hash_m, path_map, path_map_m, depth, dist, goal]
            )

    cols_to_keep = ["Depth", "Distance From Goal", "Goal"]

    if kind_of_ordering == "hash" and kind_of_data == "separated":
        col_to_add = "Path Hash"
        cols_to_keep.append(col_to_add)
    elif kind_of_ordering == "hash" and kind_of_data == "merged":
        col_to_add = ("Path Hash Merged",)
        cols_to_keep.append(col_to_add)
    elif kind_of_ordering == "map" and kind_of_data == "separated":
        col_to_add = "Path Mapped"
        cols_to_keep.append(col_to_add)
    elif kind_of_ordering == "map" and kind_of_data == "merged":
        col_to_add = "Path Mapped Merged"
        cols_to_keep.append(col_to_add)
    else:
        raise NotImplementedError(
            f"kind_of_ordering: {kind_of_ordering} - kind_of_data: {kind_of_data} not implemented"
        )

    df = pd.DataFrame(records, columns=COL_NAMES_CSV)
    df["Distance From Goal"] = pd.to_numeric(df["Distance From Goal"], errors="coerce")
    df["Depth"] = pd.to_numeric(df["Depth"], errors="coerce")

    filtered_df = df[[col for col in cols_to_keep if col in df.columns]]

    filtered_df.columns = filtered_df.columns.str.strip()

    filtered_df = filtered_df.rename(columns={col_to_add: "Path State"})

    return filtered_df


def _balance_dataset(
    df: pd.DataFrame,
    label_column: str,
    n_max_samples: int,
    max_majority_ratio: float = 0.3,
):
    # Get the value counts
    counts = df[label_column].value_counts()

    # Separate majority class
    majority_class = UNREACHABLE_STATE_VALUE
    other_classes = counts.drop(majority_class)

    # How many majority samples we can keep
    max_majority_samples = int(n_max_samples * max_majority_ratio)
    majority_df = df[df[label_column] == majority_class].sample(
        n=min(max_majority_samples, counts[majority_class]), random_state=0
    )

    # How many samples are left for other classes
    remaining_samples = n_max_samples - len(majority_df)
    n_classes = len(other_classes)

    # Determine how many samples per class (evenly, clipped by availability)
    per_class_target = remaining_samples // n_classes

    # Sample each class
    other_dfs = []
    for cls, count in other_classes.items():
        n = min(per_class_target, count)
        sampled = df[df[label_column] == cls].sample(n=n, random_state=0)
        other_dfs.append(sampled)

    # Combine all parts
    balanced_df = (
        pd.concat([majority_df] + other_dfs)
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )

    return balanced_df


def get_df(
    folder_data: str,
    kind_of_ordering: str,
    kind_of_data: str,
    max_samples_for_prob: int,
    max_unreachable_state_ratio: float = 0.3,
) -> pd.DataFrame:
    problems_folders = _get_all_items(folder_data)

    final_df = pd.DataFrame()

    for p in problems_folders:
        all_problem_items = _get_all_items(p)
        csv_file_path = next((p for p in all_problem_items if p.suffix == ".csv"), None)

        if csv_file_path is None:
            continue  # Skip if no CSV file found

        df = _read_csv(csv_file_path, kind_of_ordering, kind_of_data)

        df["Distance From Goal"] = df["Distance From Goal"].replace(
            -1, UNREACHABLE_STATE_VALUE
        )

        df = _balance_dataset(
            df,
            "Distance From Goal",
            max_samples_for_prob,
            max_majority_ratio=max_unreachable_state_ratio,
        )

        final_df = pd.concat([final_df, df], ignore_index=True)

    return final_df


def _preprocess_sample(
    state_path: str,
    depth: int,
    target: int | None = None,
    goal_path: str | None = None,
):
    """
    • Converts DOT graphs to PyG objects.
    • Packs them (plus depth / optional target) into a dict the collate_fn
      already understands.
    """

    # --------------------------------------------------------------------------- #
    #  helper ─ load a DOT file and normalise node labels                         #
    # --------------------------------------------------------------------------- #
    def _load_graph_from_dot(dot_path: str, is_goal: bool = False) -> nx.DiGraph:
        """
        Reads a .dot file into a NetworkX DiGraph.
        Goal graphs sometimes contain negative node IDs; when `is_goal=True`
        we quote them so `pydot` parses correctly.
        """
        dot_path = Path(dot_path)

        if is_goal:
            fixed_path = dot_path.with_name("fixed_" + dot_path.name)
            with open(dot_path, "r") as f:
                content = f.read()
            # quote bare negative numbers that serve as node IDs
            content = re.sub(r'(?<![\w"]) (-\d+)(?![\w"])', r'"\1"', content)
            fixed_path.write_text(content)
            dot_path = fixed_path  # parse the patched file

        G_raw = nx.DiGraph(nx.nx_pydot.read_dot(str(dot_path)))
        return nx.convert_node_labels_to_integers(G_raw, label_attribute="orig_id")

    # --------------------------------------------------------------------------- #
    #  helper ─ convert NetworkX → PyG Data with safe tensor features             #
    # --------------------------------------------------------------------------- #
    def _nx_to_pyg(
        G: nx.DiGraph,
        node_attr_keys: list[str] = ("shape",),
        edge_attr_keys: list[str] = ("label",),
    ):
        """
        • Ensures every requested attribute exists (default 0).
        • Maps *all* non-numeric values to consecutive integers.
        • Guarantees `data.x` (and `data.edge_attr`, if requested) are tensors.
        """
        # ---- ensure uniform attribute sets ------------------------------------
        for _, d in G.nodes(data=True):
            for k in node_attr_keys:
                d.setdefault(k, 0)
        for _, _, d in G.edges(data=True):
            for k in edge_attr_keys:
                d.setdefault(k, 0)

        # ---- numeric-or-categorical → int -------------------------------------
        def _to_int(val, vocab: dict):
            # strip quotes that pydot keeps around strings
            s = str(val).strip('"')
            # numeric?
            if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
                return int(float(s))
            # categorical
            if s not in vocab:
                vocab[s] = len(vocab)
            return vocab[s]

        # per-attribute vocabularies
        node_vocab = {k: {} for k in node_attr_keys}
        edge_vocab = {k: {} for k in edge_attr_keys}

        for _, d in G.nodes(data=True):
            for k in node_attr_keys:
                d[k] = _to_int(d[k], node_vocab[k])

        for _, _, d in G.edges(data=True):
            for k in edge_attr_keys:
                d[k] = _to_int(d[k], edge_vocab[k])

        # ---- basic conversion --------------------------------------------------
        data = from_networkx(
            G,
            group_node_attrs="all",
            group_edge_attrs="all",
        )  # *without* grouping → no list-of-lists bug

        # ---- build node feature matrix ----------------------------------------
        node_feats = [
            [d[k] for k in node_attr_keys] for _, d in G.nodes(data=True)
        ]  # shape (N, len(node_attr_keys))
        data.x = torch.tensor(node_feats, dtype=torch.long)

        # ---- build edge features  (optional; comment out if unused) -----------
        if edge_attr_keys and G.number_of_edges() > 0:
            edge_feats = [
                [d[k] for k in edge_attr_keys] for *_, d in G.edges(data=True)
            ]
            data.edge_attr = torch.tensor(edge_feats, dtype=torch.long)

        return data

    # state graph -----------------------------------------------------------
    G_state = _load_graph_from_dot(state_path, is_goal=False)
    data_state = _nx_to_pyg(G_state)

    sample = {
        "state_graph": data_state,
        "depth": torch.tensor([depth], dtype=torch.long),
    }

    if target is not None:
        sample["target"] = torch.tensor([target], dtype=torch.float32)

    # goal graph (optional) --------------------------------------------------
    if goal_path is not None:
        G_goal = _load_graph_from_dot(goal_path, is_goal=True)
        sample["goal_graph"] = _nx_to_pyg(G_goal)

    return sample


def load_samples(df, use_goal: bool) -> List:
    all_samples = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading samples..."):
        state_path = row["Path State"]
        goal_path = row["Goal"] if use_goal else None
        depth = int(row["Depth"])
        target = int(row["Distance From Goal"])

        # try:
        sample = _preprocess_sample(state_path, depth, target, goal_path)
        all_samples.append(sample)
        """except Exception as e:
            print(f"Error processing sample: {e}")
            continue"""

    return all_samples


def split_samples(samples, test_ratio=0.2, seed=42):  # ▶ NEW/CHANGED
    return train_test_split(
        samples, test_size=test_ratio, shuffle=True, random_state=seed, stratify=None
    )


def _graph_collate_fn(batch):
    """
    • Works for both training/eval (samples include 'target')
      and inference (no 'target').
    • Keeps API identical for the model; the training loop just checks
      if 'target' is in the batch dict before computing the loss.
    """
    collated = {
        "state_graph": Batch.from_data_list([b["state_graph"] for b in batch]),
        "goal_graph": (
            Batch.from_data_list([b["goal_graph"] for b in batch])
            if "goal_graph" in batch[0]
            else None
        ),
        "depth": torch.stack([b["depth"] for b in batch]),
    }

    if "target" in batch[0]:  # ← only in training / evaluation
        collated["target"] = torch.stack([b["target"] for b in batch])

    return collated


def _seed_everything(seed: int = 42):  # ▶ NEW/CHANGED
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    train_samples,
    eval_samples,
    batch_size=256,
    shuffle=True,
    seed=42,
    num_workers=0,
):
    # Torch RNG shared by both loaders so shuffling is reproducible
    g = torch.Generator()
    g.manual_seed(seed)

    def _build_loader(samples, is_train):
        return DataLoader(
            PrecomputedGraphDataset(samples),
            batch_size=batch_size,
            shuffle=(shuffle and is_train),
            collate_fn=_graph_collate_fn,
            num_workers=num_workers,
            generator=g,
            worker_init_fn=lambda wid: _seed_everything(seed + wid),
        )

    return _build_loader(train_samples, True), _build_loader(eval_samples, False)


class PrecomputedGraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# ░░  2.  MODEL  ░░
# ──────────────────────────────────────────────────────────────────────────────
class DistanceEstimator(nn.Module):

    def __init__(
        self,
        in_channels_node: int,
        hidden_dim: int = 64,
        use_goal: bool = True,
    ):
        super().__init__()
        self.use_goal = use_goal

        # ── GNN for the *state* graph ────────────────────
        self.state_gnn1 = SAGEConv(in_channels_node, hidden_dim)
        self.state_gnn2 = SAGEConv(hidden_dim, hidden_dim)

        # ── (optional) GNN for the *goal* graph ──────────
        if use_goal:
            self.goal_gnn1 = SAGEConv(in_channels_node, hidden_dim)
            self.goal_gnn2 = SAGEConv(hidden_dim, hidden_dim)

        # ── “head”: graph-embedding + depth  →  distance ─
        # (+1 because we append the scalar depth)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * (2 if use_goal else 1) + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # helper -----------------------------------------------------
    def _encode(self, data_batch, conv1, conv2):
        x = data_batch.x.float()
        edge_index = data_batch.edge_index
        x = F.relu(conv1(x, edge_index))
        x = F.relu(conv2(x, edge_index))
        return global_mean_pool(x, data_batch.batch)  # shape: [batch, hidden]

    # forward ----------------------------------------------------
    def forward(self, batch_dict):
        # ─ state graph ─
        state_emb = self._encode(
            batch_dict["state_graph"], self.state_gnn1, self.state_gnn2
        )

        # ─ goal graph (optional) ─
        if self.use_goal and batch_dict["goal_graph"] is not None:
            goal_emb = self._encode(
                batch_dict["goal_graph"], self.goal_gnn1, self.goal_gnn2
            )
            graph_emb = torch.cat([state_emb, goal_emb], dim=1)
        else:
            graph_emb = state_emb

        # ─ scalar depth (robust normalisation) ───────────────────────────
        depth = batch_dict["depth"].float()  # shape (B, 1)

        # Use population std (unbiased=False) so it is 0 instead of NaN for B=1
        depth_mean = depth.mean()
        depth_std = depth.std(unbiased=False)

        # If depth_std is zero (all depths identical) avoid dividing by 0
        depth = (depth - depth_mean) / (depth_std + 1e-6)

        # ─ concatenate and predict ─
        features = torch.cat([graph_emb, depth], dim=1)
        return self.mlp(features).squeeze(1)  # shape: [batch]


# ──────────────────────────────────────────────────────────────────────────────
# ░░  3.  TRAIN / VALIDATE  ░░
# ──────────────────────────────────────────────────────────────────────────────
def move_batch_to_device(batch):
    """
    Utility because our custom collate() builds a dict that mixes
    PyG Batches with ordinary tensors.
    """
    for k, v in batch.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        else:  # PyG Batch
            batch[k] = v.to(device)
    return batch


def evaluate(model, dataloader, err_threshold=0.5):
    """
    • Computes RMSE on *all* samples (including unreachable = -99999).
    • Whenever |prediction – target| > err_threshold, prints
          True=...  Pred=...
    """
    model.eval()
    mse_accum, n = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch)
            target = batch["target"].float().squeeze(1)  # (B,)
            pred = model(batch)  # (B,)

            # ── print large errors ─────────────────────────────────────────
            abs_err = (pred - target).abs()
            big_mask = abs_err > target + 0.5 * target
            if big_mask.any():
                for t_val, p_val in zip(
                    target[big_mask].tolist(), pred[big_mask].tolist()
                ):
                    print(f"True={t_val:.3f}  Pred={p_val:.3f}")

            # ── accumulate for RMSE ───────────────────────────────────────
            mse_accum += F.mse_loss(pred, target, reduction="sum").item()
            n += target.numel()

    return (mse_accum / n) ** 0.5  # global RMSE


def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs: int = 500,
    lr: float = 1e-3,
    min_delta: float = 1e-5,
    patience: int = 10,
):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_rmse = float("inf")
    patience_left = patience

    pbar = tqdm(range(n_epochs))

    past_train_rmse, past_val_rmse = 0.0, 0.0

    for _ in pbar:
        # ── 1. TRAIN ───────────────────────────────────────────────────────
        model.train()
        mse_accum, n = 0.0, 0
        for batch in train_loader:
            batch = move_batch_to_device(batch)
            target = batch["target"].float().squeeze(1)

            optim.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, target, reduction="sum")
            loss.backward()
            optim.step()

            mse_accum += loss.item()
            n += target.size(0)

        train_rmse = (mse_accum / n) ** 0.5

        # ── 2. VALIDATE ────────────────────────────────────────────────────
        val_rmse = evaluate(model, val_loader)

        pbar.set_postfix(
            delta_train_rmse=train_rmse - past_train_rmse,
            delta_val_rmse=val_rmse - past_val_rmse,
        )
        past_train_rmse, past_val_rmse = train_rmse, val_rmse

        # ── 3. EARLY-STOPPING ─────────────────────────────────────────────
        if val_rmse + min_delta < best_rmse:
            best_rmse = val_rmse
            patience_left = patience
            _save_full_checkpoint(model, best_rmse)
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break


def _save_full_checkpoint(model, best_rmse: float, path: str = "distance_estimator.pt"):
    """
    Stores model weights *and* minimal reconstruction config in one file.
    """
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "in_channels_node": model.state_gnn1.in_channels,
            "hidden_dim": model.state_gnn1.out_channels,
            "use_goal": model.use_goal,
        },
        "best_val_rmse": best_rmse,
    }
    torch.save(ckpt, path)


if __name__ == "__main__":
    FOLDER_DATA = "out/NN/Training"
    KIND_OF_ORDERING = "hash"  # "map"
    KIND_OF_DATA = "separated"  # "merged"
    USE_GOAL = True  # False

    TRAIN_SIZE = 0.75
    MAX_SAMPLES_FOR_PROB = 15000

    df_paths = get_df(FOLDER_DATA, KIND_OF_ORDERING, KIND_OF_DATA, MAX_SAMPLES_FOR_PROB)

    samples = load_samples(df_paths, USE_GOAL)

    # Save everything to a .pt file
    torch.save(
        {
            "params": {
                "FOLDER_DATA": FOLDER_DATA,
                "KIND_OF_ORDERING": KIND_OF_ORDERING,
                "KIND_OF_DATA": KIND_OF_DATA,
                "USE_GOAL": USE_GOAL,
                "MAX_SAMPLES_FOR_PROB": MAX_SAMPLES_FOR_PROB,
            },
            "df_paths": df_paths,
            "samples": samples,
        },
        "dataloader_info.pt",
    )

    samples = torch.load("dataloader_info.pt", weights_only=False)["samples"]

    train_samples, eval_samples = split_samples(samples)

    train_loader, val_loader = get_dataloaders(train_samples, eval_samples)

    # infer node-feature dimensionality from a single sample
    in_channels = train_samples[0]["state_graph"].num_node_features
    model = DistanceEstimator(
        in_channels_node=in_channels,
        hidden_dim=64,
        use_goal=USE_GOAL,
    )

    train_model(model, train_loader, val_loader)
    print("Done")
