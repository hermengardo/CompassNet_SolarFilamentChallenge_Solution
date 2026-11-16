import torch


class Config:
    seed: int = 567
    pin_memory: bool = True
    batch_size: int = 2
    resolution: int = 2048
    val_size: float = 0.1
    num_workers: int = 4
    lr: float = 1e-5
    shuffle: bool = True
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train = "/home/jaoliveira/not_related/data/supervised/magfilo/train/train_images"
    test = "/home/jaoliveira/not_related/data/supervised/magfilo/test/test_images"
    metadata = "/home/jaoliveira/not_related/data/supervised/magfilo/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    def __repr__(self) -> str:
        return f"""
        {"="*12} Config {"="*12}
        ==> Seed: {self.seed}
        ==> Device: {self.device}
        ==> Resolution: {self.resolution}
        ==> Batch size: {self.batch_size}
        ==> Num workers: {self.num_workers}
        ==> Pin memory: {self.pin_memory}
        ==> Shuffle: {self.shuffle}
        ==> Val size: {self.val_size}
        ==> LR: {self.lr}
        {"="*32}
        """

CFG = Config()
