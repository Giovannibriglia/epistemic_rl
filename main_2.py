from __future__ import annotations

import json
import os
import re
from pathlib import Path

import networkx as nx
import numpy as np

import pandas as pd
import pydot
import torch
import torch.nn.functional as F

from main import generate_simulation_name
from torch.utils.data import random_split
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, HeteroConv, SAGEConv
from tqdm import tqdm
from utils2 import predict_single

F_NODE = 1  # hashed‑ID (not degree)
F_EDGE = 1  # dummy weight
_TWO_64 = np.float64(2**64)  # constant so it is computed once


def _hash64(label) -> float:
    if int(label) == -1:
        return -1.0
    else:
        uid = np.uint64(int(label))
        return float(uid) / 2**64  # ∈ [0,1)


# --------------------------------------------------------------------
# fixed‑width node & edge feats
# --------------------------------------------------------------------
def _to_pyg_tensors(G: nx.Graph, node_to_idx: dict[int, int]):
    N = G.number_of_nodes()

    # node‑features
    x = torch.zeros((N, F_NODE), dtype=torch.float)
    x[:, 0] = torch.tensor([int(n) for n in G.nodes], dtype=torch.float)
    # x[:, 0] = torch.tensor([_hash64(n) for n in G.nodes], dtype=torch.float)
    # x[:, 1] = torch.tensor([G.degree(n) for n in G.nodes], dtype=torch.float)

    # edges
    src, dst = (
        zip(*[(node_to_idx[comb[0]], node_to_idx[comb[1]]) for comb in G.edges])
        if G.edges
        else ([], [])
    )
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # edge_attr  = torch.ones((len(src), F_EDGE), dtype=torch.float)  # weight=1
    edge_attr = torch.zeros((len(src), 1), dtype=torch.float)
    for c, (u, v, data) in enumerate(G.edges(data=True)):
        edge_attr[c] = int(data["label"].strip('"'))
    return x, edge_index, edge_attr


def _build_hetero_sample(
    G_state: nx.Graph, G_goal: nx.Graph, depth: float, dist_from_goal: float = None
):
    idx_state = {n: i for i, n in enumerate(G_state.nodes)}
    idx_goal = {n: i for i, n in enumerate(G_goal.nodes)}

    data = HeteroData()

    x_s, ei_s, ea_s = _to_pyg_tensors(G_state, idx_state)
    x_g, ei_g, ea_g = _to_pyg_tensors(G_goal, idx_goal)
    # ── node features ────────────────────────────────────────────
    data["state"].x = x_s
    data["goal"].x = x_g

    # ── homogeneous edges  state→state , goal→goal ───────────────
    data["state", "to", "state"].edge_index = ei_s
    data["state", "to", "state"].edge_attr = ea_s

    data["goal", "to", "goal"].edge_index = ei_g
    data["goal", "to", "goal"].edge_attr = ea_g

    # ── alignment edges state⇄goal (as before) ───────────────────
    src = [idx_state[n] for n in G_state.nodes if n in idx_goal]
    dst = [idx_goal[n] for n in G_state.nodes if n in idx_goal]

    def make_edges(s, d):
        ei = torch.tensor([s, d], dtype=torch.long)
        ea = torch.ones((len(s), F_EDGE), dtype=torch.float)
        if len(s) == 0:  # keep schema if empty
            ei = torch.empty((2, 0), dtype=torch.long)
            ea = torch.empty((0, F_EDGE), dtype=torch.float)
        return ei, ea

    ei_fwd, ea_fwd = make_edges(src, dst)
    ei_rev, ea_rev = make_edges(dst, src)

    data["state", "matches", "goal"].edge_index = ei_fwd
    data["state", "matches", "goal"].edge_attr = ea_fwd
    data["goal", "rev_matches", "state"].edge_index = ei_rev
    data["goal", "rev_matches", "state"].edge_attr = ea_rev

    # ── graph‑level attrs ────────────────────────────────────────
    data.depth = torch.tensor([depth], dtype=torch.float)
    if dist_from_goal is not None:
        data.y = torch.tensor([dist_from_goal], dtype=torch.float)
    return data


def _build_state_sample(
    G_state: nx.Graph, depth: float, dist_from_goal: float | None = None
):
    idx = {n: i for i, n in enumerate(G_state.nodes)}
    x, ei, ea = _to_pyg_tensors(G_state, idx)

    data = Data()
    data.x = x
    data.edge_index = ei
    data.edge_attr = ea
    data.depth = torch.tensor([depth], dtype=torch.float)
    if dist_from_goal is not None:  # 3️⃣  use the *right* variable
        data.y = torch.tensor([dist_from_goal], dtype=torch.float)
    return data


def build_sample(G_state, depth, dist=None, G_goal=None):
    if G_goal is not None:
        return _build_hetero_sample(G_state, G_goal, depth, dist)
    else:
        return _build_state_sample(G_state, depth, dist)


class BasicGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.g1 = GCNConv(-1, hidden_dim)
        self.g2 = GCNConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim + 1, 1)

    def forward(self, data):
        x = F.relu(self.g1(data.x, data.edge_index))
        x = F.relu(self.g2(x, data.edge_index))
        pooled = global_mean_pool(x, data.batch)  # [B, H]
        z = torch.cat([pooled, data.depth.unsqueeze(-1)], dim=-1)
        return self.out(z).squeeze(-1)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [
                HeteroConv(
                    {
                        ("state", "to", "state"): GCNConv(-1, hidden_dim),
                        ("goal", "to", "goal"): GCNConv(-1, hidden_dim),
                        ("state", "matches", "goal"): SAGEConv((-1, -1), hidden_dim),
                        ("goal", "rev_matches", "state"): SAGEConv(
                            (-1, -1), hidden_dim
                        ),
                    },
                    aggr="mean",
                )
                for _ in range(2)
            ]
        )
        self.out = torch.nn.Linear(2 * hidden_dim + 1, 1)

    def forward(self, data):
        x_dict, ei_dict = data.x_dict, data.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, ei_dict)
        s = global_mean_pool(x_dict["state"], data["state"].batch)  # [B,H]
        g = global_mean_pool(x_dict["goal"], data["goal"].batch)  # [B,H]
        z = torch.cat([s, g, data.depth.unsqueeze(-1)], dim=-1)
        return self.out(z).squeeze(-1)


def make_model(if_use_goal: bool):
    return HeteroGNN() if if_use_goal else BasicGNN()


def read_goal_dot(g_goal_path: Path):
    fixed_path = g_goal_path.with_name("fixed_" + g_goal_path.name)

    # Quote negative node IDs like -1
    with open(g_goal_path, "r") as f:
        content = f.read()

    # Add quotes around negative numbers only when they are node identifiers
    content_fixed = re.sub(r'(?<![\w"]) (-\d+)(?![\w"])', r'"\1"', content)

    # Save to a new file
    with open(fixed_path, "w") as f:
        f.write(content_fixed)

    (dot_graph,) = pydot.graph_from_dot_file(fixed_path)
    return nx.drawing.nx_pydot.from_pydot(dot_graph)


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TRAIN_FRAC = 0.8
    EPOCHS = 1000
    USE_HASH = input("Use hash? True/False: ").strip().lower() == "true"
    USE_GOAL = input("Use goal? True/False: ").strip().lower() == "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_data = "data/CC_3_2_3__pl_6_d_9.csv"
    dataset_name = os.path.splitext(os.path.basename(path_data))[0]
    sim_path = f"results/{generate_simulation_name(dataset_name)}"

    if USE_HASH:
        sim_path += "_hashing"
    else:
        sim_path += "_mapping"

    if USE_GOAL:
        sim_path += "_goal"
    else:
        sim_path += "_nogoal"

    os.makedirs(sim_path, exist_ok=True)

    # ── parse CSV ────────────────────────────────────────────────────
    records, COLS = [], [
        "Path Hash",
        "Path Mapped",
        "Depth",
        "Distance From Goal",
        "Goal",
    ]
    with open(path_data) as f:
        next(f)
        for raw in f:
            path_hash, path_mapped, depth, dist, goal = raw.rstrip("\n").split(",")
            records.append([path_hash, path_mapped, depth, dist, goal])

    df = pd.DataFrame(records, columns=COLS)
    # df = df[:1000]

    if USE_GOAL:
        g_goal_path = Path("./data/" + df.iloc[0, :]["Goal"].split("/")[-1])
        G_g = read_goal_dot(g_goal_path)
        # G_g = nx.DiGraph(nx.nx_pydot.read_dot(g_goal_path))
    else:
        G_g = None

    # ── build dataset ────────────────────────────────────────────────
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="building graphs..."):
        g_state_path = "./" + (row["Path Hash"] if USE_HASH else row["Path Mapped"])
        G_s = nx.DiGraph(nx.nx_pydot.read_dot(g_state_path))

        depth = float(row["Depth"])
        dist = float(row["Distance From Goal"])

        samples.append(build_sample(G_s, depth, dist, G_g))

        """if USE_GOAL:
            g_goal_path = "./data/" + row["Goal"].split("/")[-1]
            G_g = nx.DiGraph(nx.nx_pydot.read_dot(g_goal_path))
            samples.append(build_sample(G_s, depth, dist, G_g))
        else:
            samples.append(build_sample(G_s, depth, dist))"""
    # find_mismatch(samples)
    torch.save(samples, f"{sim_path}/dataset.pt")

    # ── train / val split ────────────────────────────────────────────
    dataset = torch.load(f"{sim_path}/dataset.pt", weights_only=False)
    n_train = int(TRAIN_FRAC * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)

    # ── model & optimiser ────────────────────────────────────────────
    model = make_model(USE_GOAL).to(device)
    optim = torch.optim.Adam(model.parameters())

    # ── training loop ────────────────────────────────────────────────
    model.train()
    pbar = tqdm(range(EPOCHS), desc="training model...")
    for epoch in pbar:
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    torch.save(model.state_dict(), f"{sim_path}/gnn_predictor.pt")

    # ── evaluation ───────────────────────────────────────────────────
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = torch.round(model(batch))
            preds.append(pred.cpu())
            targets.append(batch.y.view(-1).cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse = F.mse_loss(preds.float(), targets.float())
    print(f"\nEvaluation MSE: {mse.item():.4f}")

    # save results
    out = {"preds": preds.tolist(), "targets": targets.tolist()}
    with open(f"{sim_path}/predictions.json", "w") as f:
        json.dump(out, f)

    # -------------------------------------------------------------------------------------------------------
    # predict real graph
    first_row = df.iloc[0, :]
    g_state_path = "./" + (
        first_row["Path Hash"] if USE_HASH else first_row["Path Mapped"]
    )
    G_s = nx.DiGraph(nx.nx_pydot.read_dot(g_state_path))

    depth = float(first_row["Depth"])
    true_dist = float(first_row["Distance From Goal"])

    if USE_GOAL:
        g_goal_path = Path("./data/" + first_row["Goal"].split("/")[-1])
        G_g = read_goal_dot(g_goal_path)
    else:
        G_g = None

    sample = build_sample(G_s, depth, None, G_g)

    pred_dist = predict_single(sample, model, device=device)
    print("Predicted distance from goal: ", pred_dist, " --> ", int(pred_dist))
    print("True distance from goal: ", true_dist)
