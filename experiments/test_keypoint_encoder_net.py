import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from networks.keypoint_net import Image_encoder_net
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, shape_x, shape_y):
        self.shape_x = shape_x
        self.shape_y = shape_y

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        x = torch.rand(self.shape_x)
        y = torch.rand(self.shape_y)
        return x, y


class TestKeypointEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = Image_encoder_net()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        result = self.encoder(x)
        return result

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.encoder(x)
        loss = 0 #F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def do_test():
    encoder = TestKeypointEncoder()
    trainer = pl.Trainer()
    testds = TestDataset((3, 32, 32), (3, 32, 32))
    trainer.fit(encoder, DataLoader(testds), DataLoader(testds))