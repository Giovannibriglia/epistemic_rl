from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

N_EPOCHS_DEFAULT = 200


class BaseModel(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = None,
        optimizer_cls: type = torch.optim.Adam,
        optimizer_kwargs: dict = None,
    ):
        """
        Args:
            model: your nn.Module
            device: 'cuda' / 'cpu' or torch.device.  If None, auto‐selects.
            optimizer_cls: optimizer class (default Adam)
            optimizer_kwargs: dict of kwargs to pass to optimizer (e.g. {'lr':1e-3})
        """
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_epochs: int = N_EPOCHS_DEFAULT,
        checkpoint_dir: str = ".",
        **kwargs,
    ) -> None:
        """
        Generic training loop.  Subclasses must implement:
          - self._compute_loss(batch)
          - self.evaluate(loader, **eval_kwargs) → metrics_dict
          - self._save_full_checkpoint(path, **ckpt_kwargs)

        Args:
            train_loader, val_loader: usual DataLoaders
            n_epochs: total epochs
            checkpoint_dir: where to save best model
            **kwargs: passed through to evaluate()
        """
        best_metric = float("inf")
        os.makedirs(checkpoint_dir, exist_ok=True)
        pbar = tqdm(range(n_epochs), desc="training...")

        history: dict[str, list[float]] = defaultdict(list)

        for _ in pbar:
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = self._move_batch_to_device(batch)
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader, **kwargs)

            pbar.set_postfix(
                train_loss=f"{avg_loss:.4f}",
                **{k: f"{v:.4f}" for k, v in val_metrics.items()},
            )

            if val_metrics["val_loss"] < best_metric:
                best_path = f"{checkpoint_dir}/best.pt"
                self._save_full_checkpoint(best_path)

                best_metric = val_metrics["val_loss"]

            # ←–– dynamically record *all* metrics returned
            for name, value in val_metrics.items():
                history[name].append(value)

        self.save_and_plot_metrics(history, checkpoint_dir, n_epochs)

    def _move_batch_to_device(self, batch: dict) -> dict:
        for k, v in batch.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
            else:  # assume PyG Batch
                batch[k] = v.to(self.device)
        return batch

    @abstractmethod
    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Given a batch, compute and return a scalar loss Tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        **kwargs,
    ) -> dict:
        """
        Run the model in eval mode over `loader` and return a dict of metrics,
        e.g. {'val_loss': ..., 'rmse': ..., 'accuracy': ...}
        """
        raise NotImplementedError

    @abstractmethod
    def _save_full_checkpoint(self, path: str, **metrics) -> None:
        """
        Save model state_dict and any config needed, into `path`.
        You get access to all val‐metrics as well.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path_ckpt: str) -> None:
        """
        Load weights from `path_ckpt` into self.model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, batch: dict) -> torch.Tensor:
        """
        Run forward on a batch, return raw outputs or processed predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_single(self, *args, **kwargs):
        """
        Predict on a single example (e.g. file paths or tensors),
        """
        raise NotImplementedError

    @staticmethod
    def save_and_plot_metrics(history, checkpoint_dir, n_epochs):

        # — after training: plot everything you tracked —
        epochs = range(n_epochs)

        # 1) Loss curves (anything with 'loss' in name)
        plt.figure(figsize=(6, 4), dpi=500)
        for key in history:
            if "loss" in key:
                plt.plot(epochs, history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/train_loss.png")
        plt.show()

        # 2) All other metrics
        plt.figure(figsize=(6, 4), dpi=500)
        for key in history:
            if "loss" not in key:
                plt.plot(epochs, history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Metric")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{checkpoint_dir}/validation_metrics.png")
        plt.show()

        return history

    def to_onnx(self, onnx_path: str | Path, use_goal: bool = True) -> None:
        """
        Export the *trained* DistanceEstimator to ONNX with **dynamic** sizes:

            • any number of graphs     (batch axis)
            • any number of nodes/edges in each graph

        The exported model therefore runs both single-graph *and* mini-batch
        inference in ONNX Runtime.
        """
        onnx_wrapper = self._get_onnx_wrapper()

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
            onnx_wrapper(self.model).cpu(),  # << wrapped core
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

    @abstractmethod
    def _get_onnx_wrapper(self):
        raise NotImplementedError
