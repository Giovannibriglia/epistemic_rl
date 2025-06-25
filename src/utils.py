from __future__ import annotations

import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pydot
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx

from src.model import BaseModel
from src.models.distance_estimator import DistanceEstimator
from src.models.new_distance_estimator import NewDistanceEstimator
from src.models.new_reachability_classifier import NewReachabilityClassifier
from src.models.reachability_classifier import ReachabilityClassifier


class PrecomputedGraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
            collate_fn=graph_collate_fn,
            num_workers=num_workers,
            generator=g,
            worker_init_fn=lambda wid: seed_everything(seed + wid),
        )

    return _build_loader(train_samples, True), _build_loader(eval_samples, False)


def merge_rare(y, threshold=2):
    # Count labels
    from collections import Counter

    cnt = Counter(y)
    # Any label with fewer than `threshold` occurrences becomes “other”
    return ["other" if cnt[label] < threshold else label for label in y]


def split_samples(samples, test_ratio=0.2, seed=42):
    # Extract the list of labels
    y = [sample["target"] for sample in samples]
    y_merged = merge_rare(y, threshold=2)

    # Now split, stratifying on y
    train, test = train_test_split(
        samples,
        test_size=test_ratio,
        shuffle=True,
        random_state=seed,
        stratify=y_merged,
    )
    return train, test


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_NEG_ID_RE = re.compile(r'(?<![\w"])-\d+(?![\w"])')


def _load_dot(path: Path, quote_neg: bool) -> nx.DiGraph:
    src = path.read_text()
    if quote_neg:
        src = _NEG_ID_RE.sub(lambda m: f'"{m.group(0)}"', src)
    dot = pydot.graph_from_dot_data(src)[0]
    return nx.nx_pydot.from_pydot(dot)


def plot_graph(G: nx.Graph):
    assert isinstance(G, nx.MultiDiGraph)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=800)
    nx.draw_networkx_labels(G, pos)
    for i, (u, v, k, d) in enumerate(G.edges(keys=True, data=True)):
        rad = 0.2 * (k + 1)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
            arrows=True,
        )
        x, y = (pos[u] + pos[v]) / 2
        plt.text(x, y + rad, str(d.get("label", "")), fontsize=9, color="red")
    plt.axis("off")
    plt.show()


def diagnose_data(data):
    print("\n=== PyG Graph Diagnostics ===")

    # Node info
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Node indices: {list(range(data.num_nodes))}")
    print(f"Node names: {data.node_names}")
    if hasattr(data, "x"):
        print(f"Node features - shape node (x):\n{data.x}")
    else:
        print("No node features ('x') set.")

    # Edge info
    print(f"Number of edges: {data.num_edges}")
    print(f"Edge index (source -> target):\n{data.edge_index}")

    # Edge labels
    if hasattr(data, "edge_attr"):
        print(f"Edge attributes (labels):\n{data.edge_attr.squeeze()}")
    else:
        print("No edge attributes ('edge_attr') set.")

    # Edge labels
    if hasattr(data, "edge_type"):
        print(f"Edge type :\n{data.edge_type.squeeze()}")
    else:
        print("No edge type ('edge_type') set.")

    # Check graph consistency
    src_nodes = data.edge_index[0].tolist()
    tgt_nodes = data.edge_index[1].tolist()
    edge_labels = data.edge_attr.squeeze().tolist()

    print("--- Edge Summary ---")
    for i, (src, tgt, label) in enumerate(zip(src_nodes, tgt_nodes, edge_labels)):
        print(f"Edge {i}: {src} -> {tgt} | Label: {label}")


def _nx_to_pyg(G: nx.DiGraph, plot: bool = False, diagnose: bool = False) -> Data:
    for n, data in G.nodes(data=True):
        data["shape"] = {"circle": 0, "doublecircle": 1}.get(
            data.get("shape", "circle"), 0
        )
    for u, v, d in G.edges(data=True):
        d["edge_label"] = int(str(d.get("label", "0")).replace('"', ""))
    # optional plotting
    if plot:
        plot_graph(G)

    # convert to a PyG Data object
    data = from_networkx(G)

    # 1) Directed edges only; no duplication
    # Edge labels come from d["label"]
    edge_labels = data.edge_label.view(-1, 1).float()  # [E,1]
    data.edge_index = data.edge_index.long()
    data.edge_attr = edge_labels

    # 2) Node IDs → float tensor
    raw_ids = [int(n) for n in G.nodes()]
    data.node_names = torch.tensor(raw_ids, dtype=torch.float)

    # 3) Clean up unused fields if you like
    # del data.edge_label, data.edge_type, data.x

    if diagnose:
        diagnose_data(data)

    return data


def preprocess_sample(
    state_path: str,
    depth: int,
    target: int | None = None,
    goal_path: Optional[str] = None,
    if_plot_graph: bool = False,
    if_diagnose: bool = False,
) -> Dict[str, Any]:
    fix = "merged" in state_path
    Gs = _load_dot(Path(state_path), fix)
    if if_plot_graph:
        plot_graph(Gs)
    ds = _nx_to_pyg(Gs)
    assert ds.edge_index.size(1) == ds.edge_attr.size(
        0
    ), f"Mismatch: {ds.edge_index.size(1)} edges vs {ds.edge_attr.size(0)} attrs"

    if if_diagnose:
        diagnose_data(ds)
    ds.name = Path(state_path).stem

    sample = {"state_graph": ds, "depth": torch.tensor([depth])}

    if target is not None:
        sample["target"] = torch.tensor([target], dtype=torch.float)

    if goal_path is not None:
        Gg = _load_dot(Path(goal_path), True)
        if if_plot_graph:
            plot_graph(Gg)
        dg = _nx_to_pyg(Gg)
        assert dg.edge_index.size(1) == dg.edge_attr.size(
            0
        ), f"Mismatch: {dg.edge_index.size(1)} edges vs {dg.edge_attr.size(0)} attrs"
        if if_diagnose:
            diagnose_data(dg)
        dg.name = Path(goal_path).stem
        sample["goal_graph"] = dg

    return sample


def graph_collate_fn(batch):
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


_TWO48_MINUS1 = float(2**48 - 1)  # for the ID normaliser


# 2) Build a thin ONNX-friendly wrapper
class OnnxWrapper(nn.Module):
    """
     Wraps `NewDistanceEstimator` so that ONNX sees only *plain tensors* –
     not PyG Data objects.  The wrapper re-implements _encode() for raw
     tensors, including a scatter-based global-mean-pool that is 100 % ONNX
    -export-able.
    """

    def __init__(self, core: "NewDistanceEstimator"):
        super().__init__()
        self.core = core  # the trained model √

    # --------------------------------------------------------------------- util
    @staticmethod
    def _global_mean(x, batch):
        """
                x     : [N, F]       (node embeddings)
                batch : [N]  (graph-id per node, 0 … B-1)
        1
                Pure-Torch implementation, ONNX-friendly, **no external scatter lib**.
        """
        # 1) accumulate sums per graph  ────────────────────────────────
        B = batch.max() + 1  # <— remains a Tensor!
        sums = torch.zeros(B, x.size(1), dtype=x.dtype, device=x.device)
        sums = sums.index_add(0, batch, x)  # [B, F]

        # 2) node counts per graph  ───────────────────────────────────
        ones = torch.ones_like(batch, dtype=x.dtype)
        cnts = torch.zeros(B, 1, dtype=x.dtype, device=x.device)
        cnts = cnts.index_add(0, batch, ones.unsqueeze(1))  # [B, 1]

        # 3) mean = sum / count  ( +ε avoids div-by-0 for empty graphs )
        return sums / cnts.clamp(min=1.0)

    # ----------------------------------------------------------------- encoder
    def _encode_raw(
        self, node_ids, edge_index, edge_attr, batch, conv1: nn.Module, conv2: nn.Module
    ):
        """
        A *tensor-only* version of NewDistanceEstimator._encode().
        Shapes are identical to the tensors fed from ONNX.
        """
        # 1) node scalar → [N, node_emb_dim]
        norm = torch.clamp((node_ids.float() + 2.0) / _TWO48_MINUS1, 0.0, 1.0)
        x = self.core.id_mlp(norm.unsqueeze(1))

        # 2) edge scalar → [E, edge_emb_dim]
        e = self.core.edge_mlp(edge_attr.float())

        # 3) two GINEConv layers
        x = F.relu(conv1(x, edge_index, e))
        x = F.relu(conv2(x, edge_index, e))

        # 4) global mean pool → [B, hidden_dim]
        return self._global_mean(x, batch)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        s_node_ids,
        s_edge_index,
        s_edge_attr,
        s_batch,
        depth,
        g_node_ids,
        g_edge_index,
        g_edge_attr,
        g_batch,
    ):
        """
        *Exactly* the signature you will export / feed from ONNX.
        """
        # ---------- state graph ----------
        s_emb = self._encode_raw(
            s_node_ids,
            s_edge_index,
            s_edge_attr,
            s_batch,
            self.core.state_conv1,
            self.core.state_conv2,
        )
        # ---------- goal graph (can be empty) ----------
        if self.core.use_goal and g_node_ids.numel() > 0:
            g_emb = self._encode_raw(
                g_node_ids,
                g_edge_index,
                g_edge_attr,
                g_batch,
                self.core.goal_conv1,
                self.core.goal_conv2,
            )
            rep = torch.cat([s_emb, g_emb], dim=1)  # [B, 2·H]
        else:
            rep = s_emb  # [B,   H]

        z = torch.cat([rep, depth.float().view(-1, 1)], dim=1)  # + depth
        return self.core.regressor(z).squeeze(1)  # → [B]


class DistanceEstimatorModel(BaseModel):
    def __init__(
        self,
        estimator_cls,  # ← a class, e.g. DistanceEstimator or ReachabilityClassifier
        *estimator_args,
        **estimator_kwargs,
    ):
        # instantiate whatever class you passed in:
        self.estimator_cls = estimator_cls(*estimator_args, **estimator_kwargs)

        # now call your base initializer
        super().__init__(model=self.estimator_cls, optimizer_kwargs={"lr": 1e-3})
        self.criterion = nn.MSELoss()

    def _compute_loss(self, batch):
        preds = self.model(batch)  # [B]
        targets = batch["target"].view(-1)  # [B]
        return self.criterion(preds, targets)

    def evaluate(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict:
        """
        Basic regression evaluation over all samples.
        Returns:
            - val_loss: mean squared error over entire set
            - mse: same as val_loss
            - rmse: square root of mse
            - mae: mean absolute error
            - r2: R² score
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                preds = self.model(batch).view(-1).cpu().tolist()
                targets = batch["target"].view(-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(targets)

        # now compute metrics via sklearn
        mse = mean_squared_error(all_targets, all_preds)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        return {
            "val_loss": mse,  # MSE as the main loss
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def _save_full_checkpoint(self, path, **metrics):
        """
        Save model + config + metrics into one file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = self.model.get_checkpoint()
        ckpt["metrics"] = metrics
        torch.save(ckpt, path)

    def load_model(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location=self.device)
        self.model = self.estimator_cls.load_model(ckpt)
        self.model.to(self.device)

    def predict_batch(self, batch):
        self.model.eval()
        batch = self._move_batch_to_device(batch)
        with torch.no_grad():
            preds = self.model(batch)
        return preds.cpu()

    def predict_single(self, state_dot: str, depth: int, goal_dot: str | None = None):
        """
        Wraps the preprocessing and returns a dict with:
          - predicted_distance: float
        """
        # 1) preprocess
        sample = preprocess_sample(
            state_path=state_dot,
            depth=depth,
            target=None,
            goal_path=goal_dot,
        )
        batch = graph_collate_fn([sample])
        # 2) predict
        pred = self.predict_batch(batch).item()
        return pred

    def to_onnx(self, onnx_path: str | Path) -> None:
        """
        Export the *trained* DistanceEstimator to ONNX with **dynamic** sizes:

            • any number of graphs     (batch axis)
            • any number of nodes/edges in each graph

        The exported model therefore runs both single-graph *and* mini-batch
        inference in ONNX Runtime.
        """
        onnx_path = Path(onnx_path)
        self.model.eval()

        # dummy – just to trace the graph; real sizes don’t matter
        N_s, E_s = 3, 4  # state  graph:  3 nodes – 4 edges
        N_g, E_g = 3, 4  # goal   graph:  3 nodes – 4 edges
        B = 2  # two graphs in the mini-batch

        dummy_inputs = (
            torch.arange(N_s, dtype=torch.float32),  # state_node_names
            torch.zeros((2, E_s), dtype=torch.int64),  # state_edge_index
            torch.zeros((E_s, 1), dtype=torch.float32),  # state_edge_attr
            torch.tensor([0, 0, 1], dtype=torch.int64),  # state_batch
            torch.zeros(B, dtype=torch.float32),  # depth  (B × 1)
            torch.arange(N_g, dtype=torch.float32),  # goal_node_names
            torch.zeros((2, E_g), dtype=torch.int64),  # goal_edge_index
            torch.zeros((E_g, 1), dtype=torch.float32),  # goal_edge_attr
            torch.tensor([0, 0, 1], dtype=torch.int64),  # goal_batch
        )

        torch.onnx.export(
            OnnxWrapper(self.model).cpu(),  # << wrapped core
            dummy_inputs,
            onnx_path.as_posix(),
            opset_version=18,  # ScatterND w/ reduction
            input_names=[
                "state_node_names",
                "state_edge_index",
                "state_edge_attr",
                "state_batch",
                "depth",
                "goal_node_names",
                "goal_edge_index",
                "goal_edge_attr",
                "goal_batch",
            ],
            output_names=["distance"],
            dynamic_axes={  # ← make everything var-len
                "state_node_names": {0: "Ns"},
                "state_edge_index": {1: "Es"},
                "state_edge_attr": {0: "Es"},
                "state_batch": {0: "Ns"},
                "depth": {0: "B"},  # B = batch size
                "goal_node_names": {0: "Ng"},
                "goal_edge_index": {1: "Eg"},
                "goal_edge_attr": {0: "Eg"},
                "goal_batch": {0: "Ng"},
                "distance": {0: "B"},
            },
        )
        print("ONNX export complete ✅")

    def try_onnx(
        self,
        onnx_path: str | Path,
        state_dot_files: Sequence[str | Path],
        depths: Sequence[float | int],
        goal_dot_files: Optional[Sequence[str | Path]] = None,
    ) -> np.ndarray:
        """
        Run a **batch** of (state, goal, depth) samples through ONNX Runtime.

        Parameters
        ----------
        state_dot_files : list[str]   – DOT files of *state* graphs
        depths          : list[int]   – raw depth values (same length as batch)
        goal_dot_files  : list[str] | None – optional DOT files of *goal* graphs
        """
        import onnxruntime as ort
        from torch_geometric.utils import from_networkx

        def _parse_dot(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """read DOT → PyG tensors (node-ids, edge_index, edge_attr)"""

            G = _load_dot(path, True)
            for n, d in G.nodes(data=True):
                d["shape"] = {"circle": 0, "doublecircle": 1}.get(
                    d.get("shape", "circle"), 0
                )
            for u, v, d in G.edges(data=True):
                d["edge_label"] = int(str(d.get("label", "0")).strip('"'))

            data = from_networkx(G)
            node_ids = torch.tensor([int(x) for x in G.nodes()], dtype=torch.float32)
            return node_ids, data.edge_index.long(), data.edge_label.view(-1, 1).float()

        # ------------- build a *concatenated* big graph --------------------
        #
        #    Each sample keeps its own graph-id via the *_batch* arrays;
        #    edge-indices are offset so that node indexing stays correct.
        #
        s_nodes, s_edges, s_attrs, s_batch = [], [], [], []
        g_nodes, g_edges, g_attrs, g_batch = [], [], [], []

        cum_state = 0
        cum_goal = 0
        for g_idx, s_file in enumerate(state_dot_files):
            n_id, e_idx, e_attr = _parse_dot(Path(s_file))
            s_nodes.append(n_id)
            s_edges.append(e_idx + cum_state)  # offset
            s_attrs.append(e_attr)
            s_batch.append(torch.full((n_id.size(0),), g_idx, dtype=torch.int64))
            cum_state += n_id.size(0)

        if goal_dot_files is not None:
            print(goal_dot_files)
            for g_idx, g_file in enumerate(goal_dot_files):
                print(g_idx, g_file)
                n_id, e_idx, e_attr = _parse_dot(Path(g_file))
                g_nodes.append(n_id)
                g_edges.append(e_idx + cum_goal)  # offset
                g_attrs.append(e_attr)
                g_batch.append(torch.full((n_id.size(0),), g_idx, dtype=torch.int64))
                cum_goal += n_id.size(0)

        # concatenate everything -------------------------------------------------
        state_node_names = torch.cat(s_nodes)  # [Ns]
        state_edge_index = torch.cat(s_edges, dim=1)  # [2,Es]
        state_edge_attr = torch.cat(s_attrs)  # [Es,1]
        state_batch = torch.cat(s_batch)  # [Ns]

        if goal_dot_files is not None:
            goal_node_names = torch.cat(g_nodes)  # [Ng]
            goal_edge_index = torch.cat(g_edges, dim=1)  # [2,Eg]
            goal_edge_attr = torch.cat(g_attrs)  # [Eg,1]
            goal_batch = torch.cat(g_batch)  # [Ng]
        else:  # empty goal
            goal_node_names = torch.empty(0, dtype=torch.float32)
            goal_edge_index = torch.empty((2, 0), dtype=torch.int64)
            goal_edge_attr = torch.empty((0, 1), dtype=torch.float32)
            goal_batch = torch.empty(0, dtype=torch.int64)

        depth = torch.as_tensor(depths, dtype=torch.float32)  # [B]

        # feed-dict ---------------------------------------------------------------
        feed = {
            "state_node_names": state_node_names.numpy(),
            "state_edge_index": state_edge_index.numpy(),
            "state_edge_attr": state_edge_attr.numpy(),
            "state_batch": state_batch.numpy(),
            "depth": depth.numpy(),
            "goal_node_names": goal_node_names.numpy(),
            "goal_edge_index": goal_edge_index.numpy(),
            "goal_edge_attr": goal_edge_attr.numpy(),
            "goal_batch": goal_batch.numpy(),
        }

        # inference ---------------------------------------------------------------
        ort_sess = ort.InferenceSession(str(onnx_path))
        distance = ort_sess.run(None, feed)[0]  # → np.ndarray  shape [B]
        return distance


class ReachabilityClassifierModel(BaseModel):
    def __init__(
        self,
        classifier_cls,  # ← a class, e.g. DistanceEstimator or ReachabilityClassifier
        *estimator_args,
        **estimator_kwargs,
    ):
        # instantiate whatever class you passed in:
        self.classifier_cls = classifier_cls(*estimator_args, **estimator_kwargs)
        # now call your base initializer
        super().__init__(model=self.classifier_cls, optimizer_kwargs={"lr": 1e-3})
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def _compute_loss(self, batch):
        logits = self.model(batch).view(-1)
        targets = batch["target"].view(-1).float()  # assume 0/1
        return self.criterion(logits, targets)

    def evaluate(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict:
        """
        Evaluate a binary reachability classifier.
        Assumes batch["target"] is already 0/1 (float or long).
        Returns:
          - val_loss: average BCE loss over all samples
          - accuracy, precision, recall, f1: classification metrics
        """
        self.model.eval()
        loss_accum = 0.0
        n = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                logits = self.model(batch).view(-1)  # raw scores
                targets = batch["target"].view(-1).float()  # 0.0 or 1.0

                # sum‐reduction BCE
                loss_accum += F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="sum"
                ).item()
                n += targets.size(0)

                # compute predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().long().tolist())

        val_loss = loss_accum / n if n > 0 else float("nan")
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)

        return {
            "val_loss": val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _save_full_checkpoint(self, path, **metrics):
        """
        Save model + config + metrics into one file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = self.model.get_checkpoint()
        ckpt["metrics"] = metrics
        torch.save(ckpt, path)

    def load_model(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location=self.device)
        self.model = self.classifier_cls.load_model(ckpt)
        self.model.to(self.device)

    def predict_batch(self, batch: dict) -> torch.Tensor:
        self.model.eval()
        batch = self._move_batch_to_device(batch)
        with torch.no_grad():
            logits = self.model(batch).view(-1)
            return torch.sigmoid(logits).cpu()  # returns probabilities

    def predict_single(
        self,
        state_dot: str,
        depth: int,
        goal_dot: str | None = None,
        threshold: float = 0.5,
    ) -> dict:
        """
        Preprocess one sample and return:
          - prob_reachable: float
          - reachable: bool
        """
        sample = preprocess_sample(
            state_path=state_dot, depth=depth, target=None, goal_path=goal_dot
        )
        batch = graph_collate_fn([sample])
        prob = self.predict_batch(batch).item()
        return {"prob_reachable": prob, "reachable": prob >= threshold}

    def to_onnx(self, path_file):
        pass


def select_model(model_name: str = "distance_estimator", use_goal: bool = True):

    if model_name == "distance_estimator":
        model = DistanceEstimatorModel(
            DistanceEstimator,
            in_channels_node=1,
            hidden_dim=64,
            use_goal=use_goal,
            num_relations=64,
        )
        return model

    elif model_name == "reachability_classifier":
        model = ReachabilityClassifierModel(
            ReachabilityClassifier,
            in_channels_node=1,
            hidden_dim=64,
            use_goal=use_goal,
            num_relations=64,
        )
        return model

    elif model_name == "new_distance_estimator":
        model = DistanceEstimatorModel(
            NewDistanceEstimator,
            use_goal=use_goal,
        )
        return model

    elif model_name == "new_reachability_classifier":
        model = ReachabilityClassifierModel(
            NewReachabilityClassifier,
            use_goal=use_goal,
        )
        return model

    else:
        raise NotImplementedError
