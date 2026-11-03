"""Dataset HAM10000 prétraité (M1) : à importer dans le code d'entraînement.

Utilisation :
    from ham_dataset import make_loaders
    train_dl, val_dl, test_dl = make_loaders("splits.csv", "ham_pretraite", img_size=224)
"""
import cv2, numpy as np, pandas as pd, torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def _crop(size):
    try:    return A.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0))
    except TypeError: return A.RandomResizedCrop(size, size, scale=(0.75, 1.0))


def _rotation():
    """Rotation de -30° à +30°, coins remplis par effet miroir (pas de coins noirs)."""
    try:    return A.Affine(rotate=(-30, 30), border_mode=cv2.BORDER_REFLECT_101, p=0.5)
    except TypeError: return A.Affine(rotate=(-30, 30), mode=cv2.BORDER_REFLECT_101, p=0.5)


def get_transforms(img_size=224):
    train = A.Compose([
        _crop(img_size), A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
        _rotation(),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.7),
        A.Normalize()])
    evaluation = A.Compose([A.Resize(img_size, img_size), A.Normalize()])
    return train, evaluation


class HAMDataset(Dataset):
    def __init__(self, frame, image_dir, transform):
        self.ids = frame["image_id"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.image_dir = str(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(f"{self.image_dir}/{self.ids[i]}.jpg"), cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]                               # H x W x C
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))   # C x H x W
        return img, self.labels[i]


def make_loaders(splits_csv, image_dir, img_size=224, batch_size=32,
                 num_workers=2, balanced=True):
    """Renvoie (train, val, test). Si balanced=True, le train est rééchantillonné
    pour que chaque classe soit vue aussi souvent (WeightedRandomSampler)."""
    df = pd.read_csv(splits_csv)
    train_tf, eval_tf = get_transforms(img_size)
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}

    train_ds = HAMDataset(parts["train"], image_dir, train_tf)
    if balanced:
        counts = parts["train"]["label"].value_counts()
        weights = parts["train"]["label"].map(lambda c: 1.0 / counts[c]).values
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)
        train_dl = DataLoader(train_ds, batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)

    val_dl = DataLoader(HAMDataset(parts["val"], image_dir, eval_tf), batch_size, num_workers=num_workers)
    test_dl = DataLoader(HAMDataset(parts["test"], image_dir, eval_tf), batch_size, num_workers=num_workers)
    return train_dl, val_dl, test_dl
