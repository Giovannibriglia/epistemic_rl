from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool, RGCNConv


class DistanceEstimator(nn.Module):
    def __init__(
        self,
        in_channels_node: int,
        hidden_dim: int = 64,
        use_goal: bool = True,
        num_relations: int = 2**5,
    ):
        super().__init__()
        self.use_goal = use_goal

        self.state_gnn1 = RGCNConv(in_channels_node, hidden_dim, num_relations)
        self.state_gnn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        if use_goal:
            self.goal_gnn1 = RGCNConv(in_channels_node, hidden_dim, num_relations)
            self.goal_gnn2 = RGCNConv(hidden_dim, hidden_dim, num_relations)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * (2 if use_goal else 1) + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, data_batch, conv1, conv2):
        x = data_batch.x.float()
        edge_index = data_batch.edge_index
        edge_type = data_batch.edge_type
        x = F.relu(conv1(x, edge_index, edge_type))
        x = F.relu(conv2(x, edge_index, edge_type))
        return global_mean_pool(x, data_batch.batch)

    def forward(self, batch_dict):
        state_emb = self._encode(
            batch_dict["state_graph"], self.state_gnn1, self.state_gnn2
        )

        if self.use_goal and batch_dict.get("goal_graph") is not None:
            goal_emb = self._encode(
                batch_dict["goal_graph"], self.goal_gnn1, self.goal_gnn2
            )
            graph_emb = torch.cat([state_emb, goal_emb], dim=1)
        else:
            graph_emb = state_emb

        depth = batch_dict["depth"].float().view(-1, 1)
        depth = (depth - depth.mean()) / (depth.std(unbiased=False) + 1e-6)

        z = torch.cat([graph_emb, depth], dim=1)
        pred = self.regressor(z).squeeze(1)
        return pred

    def get_checkpoint(self) -> Dict:
        cfg = {
            "in_channels_node": self.model.state_gnn1.in_channels,
            "hidden_dim": self.model.state_gnn1.out_channels,
            "use_goal": self.model.use_goal,
            "num_relations": self.model.state_gnn1.num_relations,
        }
        ckpt = {
            "state_dict": self.model.state_dict(),
            "config": cfg,
        }

        return ckpt

    @classmethod
    def load_model(cls, ckpt: Dict):
        cfg = ckpt["config"]
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
