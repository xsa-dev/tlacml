import argparse
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from light.data import SignMNISTDataModule
from light.module import SignLanguageModel


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
