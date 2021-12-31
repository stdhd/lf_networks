from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,WeightedRandomSampler
import numpy as np
from data import get_dataset
from data.samplers import FixedLengthSampler
from torch.utils.data import BatchSampler,RandomSampler,SequentialSampler, WeightedRandomSampler
class StaticDataModule(LightningDataModule):
    def __init__(self, config, datakeys, debug=False, custom_transforms=None):
        from data.flow_dataset import IperDataset
        super().__init__()
        self.config = config
        self.datakeys = datakeys
        self.batch_size = self.config["optimization"]["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.dset, self.transforms = get_dataset(self.config['dataset'], custom_transforms=custom_transforms)
        self.dset_train = self.dset(self.transforms, self.datakeys, self.config['dataset'], train=True, debug=debug)

        self.test_datakeys = self.datakeys

        self.dset_val = self.dset(self.transforms, self.test_datakeys, self.config['dataset'], train=False, debug=debug)
        #self.val_obj_weighting = self.config['object_weighting'] if 'object_weighting' in self.config else self.dset_val.obj_weighting

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        #sampler = SequentialSampler(self.dset_train)
        return DataLoader(self.dset_train, batch_size=self.config['optimization']['batch_size'], worker_init_fn=self.worker_init_fn, drop_last=True, num_workers=self.config['num_workers'], shuffle=True, pin_memory=False)

    def val_dataloader(self):
        #sampler = SequentialSampler(self.dset_train
        return DataLoader(self.dset_val, batch_size=self.config['optimization']['batch_size'], worker_init_fn=self.worker_init_fn, drop_last=True, num_workers=min(self.config['num_workers'] // 3, 1))

    def test_dataloader(self):
        return DataLoader(self.dset_val, batch_size=self.config['optimization']['batch_size'], shuffle=False, worker_init_fn=self.worker_init_fn, drop_last=True, num_workers=min(self.config['num_workers'] // 3, 1))
