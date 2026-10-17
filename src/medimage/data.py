from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

DATA_FLAG = "pneumoniamnist"

class MedMNISTTorchDataset(Dataset):
    def __init__(self, split="train", download=True):
        info = INFO[DATA_FLAG]
        dataset_cls = getattr(medmnist, info["python_class"])
        data_root = Path("data") / "medmnist"
        data_root.mkdir(parents=True, exist_ok=True)

        self.dataset = dataset_cls(
            split=split,
            download=download,
            root=str(data_root)
        )
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        label = int(np.asarray(label).reshape(-1)[0])
        return image, torch.tensor(label, dtype=torch.long)

def make_loader(split, batch_size=64, shuffle=False, num_workers=0, download=True):
    ds = MedMNISTTorchDataset(split=split, download=download)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
