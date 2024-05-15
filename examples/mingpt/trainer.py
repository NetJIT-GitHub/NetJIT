"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import datetime
import os
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

import fsspec
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None


@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int


class Trainer:

    def __init__(self, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.save_every = self.config.save_every
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )
        # save snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.config.snapshot_path)

        print(f"Snapshot saved at epoch {epoch}")

    def train(self):
        with open("step.txt", "w") as f:
            f.truncate(0)
        for epoch in range(self.epochs_run, self.config.max_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            for iter, (source, targets) in enumerate(self.train_loader):
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
                with torch.set_grad_enabled(True), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.config.use_amp):
                    _, loss = self.model(source, targets)

                self.optimizer.zero_grad(set_to_none=True)
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    self.optimizer.step()

                batch_loss = loss.item()
                if iter % 100 == 0:
                    print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | Train Loss {batch_loss:.5f} | Time {datetime.datetime.now()}")
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
