from pytorch_lightning import LightningModule
from torch import nn, optim


class SignLanguageModel(LightningModule):
    def __init__(self, num_classes=24, stride=1, dilation=1):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.n_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        self.lin1 = nn.Linear(in_features=16 * 7 * 7, out_features=100)
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
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3, verbose=True
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
