from pytorch_lightning import LightningDataModule
import torchvision.transforms as transforms
import torch.utils.data as data


class SignMNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=200):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=(-180, 180))], p=0.2
                ),
            ]
        )

    def prepare_data(self):
        # Data loading (called only on 1 GPU)
        import pandas as pd

        self.train_df = pd.read_csv("./data//sign_mnist_train.csv")
        self.test_df = pd.read_csv("./data//sign_mnist_test.csv")

    def setup(self, stage=None):
        # Split into train/val/test (called on every GPU)
        train_df = self.train_df.sample(frac=0.8, random_state=42)
        val_df = self.train_df.drop(train_df.index)

        self.train_dataset = SignMNISTDataset(train_df, self.transform)
        self.val_dataset = SignMNISTDataset(val_df, self.transform)
        self.test_dataset = SignMNISTDataset(self.test_df, self.transform)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )


class SignMNISTDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.labels = df["label"].values
        self.images = (
            df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        )
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
