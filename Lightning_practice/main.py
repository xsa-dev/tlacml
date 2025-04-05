import argparse
import torch
from torch import nn, optim
import torch.utils.data as data
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as transforms


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


class SignLanguageModel(LightningModule):
    def __init__(self, num_classes=24, stride=1, dilation=1):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.n_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=self.stride, dilation=self.dilation),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=self.stride, dilation=self.dilation),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.lin1 = nn.Linear(in_features=16*7*7, out_features=100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            ),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]


class SignMNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=200):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180))], p=0.2)
            ]
        )

    def prepare_data(self):
        # Data loading (called only on 1 GPU)
        import pandas as pd

        self.train_df = pd.read_csv("../data/sign_mnist_train.csv")
        self.test_df = pd.read_csv("../data/sign_mnist_test.csv")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    args = parser.parse_args()

    datamodule = SignMNISTDataModule()
    model = SignLanguageModel()

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        filename="best-model-{epoch:02d}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=10,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint_callback],
    )

    try:
        trainer.fit(model, datamodule)
        if args.fast_dev_run:
            print("Test run completed successfully")
        # Save model weights
        torch.save(model.state_dict(), "final_model_weights.pth")
        print("Model weights saved in final_model_weights.pth")

    except Exception as e:
        if args.fast_dev_run:
            print("Test run failed with an error")
            return
        raise e

    test_loader = datamodule.test_dataloader()
    test_sample = next(iter(test_loader))[0][0].unsqueeze(0)
    prediction = torch.argmax(model(test_sample), dim=1)
    print(f"Prediction for test sample: {prediction.item()}")


if __name__ == "__main__":
    main()
