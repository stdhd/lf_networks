from data.base_dataset import BaseDataset
from torchvision import transforms as tt
from data.flow_dataset import PlantDataset, IperDataset,Human36mDataset, VegetationDataset, LargeVegetationDataset, BairDataset, TaichiDataset, SharedDataset
from data.samplers import SequenceSampler,FixedLengthSampler,SequenceLengthSampler
from data.imagenet import ImageNetDataset
from data.ci_dataset import CIH36mDataset,CITaichiDataset,CIPlantDataset,CIIperDataset
from data.bair.bair import BairDataset
#from data.toy_dataset import ToyDataset

# add key value pair for datasets here, all datasets should inherit from base_dataset
__datasets__ = {"IperDataset": IperDataset,
                "PlantDataset": PlantDataset,
                "Human36mDataset": Human36mDataset,
                "VegetationDataset": VegetationDataset,
                "LargeVegetationDataset": LargeVegetationDataset,
                "BairDataset": BairDataset,
                "TaichiDataset": TaichiDataset,
                "CIPlants": CIPlantDataset,
                "CIper": CIIperDataset,
                "CITaichi": CITaichiDataset,
                "CIH36m": CIH36mDataset,
                "Bair": BairDataset}
              #  "ToyDataset": ToyDataset}

__samplers__ = {"fixed_length": FixedLengthSampler,
                }

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# returns only the class, not yet an instance
def get_transforms(config):
    return {
        "PlantDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "IperDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "Human36mDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "VegetationDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "LargeVegetationDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "BairDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "TaichiDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "CIPlants": tt.Compose(
            [
                tt.ToTensor(),
            ]),
        "CIper": tt.Compose(
            [
                tt.ToTensor(),
            ]),
        "CITaichi": tt.Compose(
            [
                tt.ToTensor(),
            ]),
        "CIH36m": tt.Compose(
            [
                tt.ToTensor(),
            ]),
        "ToyDataset": tt.Compose(
            [
                tt.ToTensor(),
            ]),
        "Bair": tt.Compose(
            [
                tt.ToTensor(),
                #tt.Lambda(lambda x: (x * 2.0) - 1.0),
                #tt.RandomHorizontalFlip(p=0.5),
                #tt.RandomVerticalFlip(p=0.5)
            ]
        ),
    }


def get_dataset(config, custom_transforms=None):
    dataset = __datasets__[config["name"]]
    if custom_transforms is not None:
        print("Returning dataset with custom transform")
        transforms = custom_transforms
    else:
        transforms = get_transforms(config)[config["name"]]
    return dataset, transforms

