


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from cvtorchvision import cvtransforms
from pathlib import Path
import os
import rasterio
import cv2


EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B8A']
RGB_BANDS = ['B04', 'B03', 'B02']

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B10': 10,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B10': 1,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}




def is_valid_file(filename):
    return filename.lower().endswith(EXTENSIONS)


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img





class EurosatDataset(Dataset):

    def __init__(self, root, split=None, bands='B13', transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        if bands=='B13':
            self.bands = ALL_BANDS

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        self.targets = []

        for froot, _, fnames in sorted(os.walk(root, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(froot, fname)
                    self.samples.append(path)
                    target = self.class_to_idx[Path(path).parts[-2]]
                    self.targets.append(target)



    def __getitem__(self, index):
        path = self.samples[index]
        target = self.targets[index]
        
        with rasterio.open(path) as f:
            array = f.read().astype(np.int32)
            img = array.transpose(1, 2, 0)

        channels = []
        for i,b in enumerate(self.bands):
            if b=='B10':
                continue
            else:
                ch = img[:,:,i]
                ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                #ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                
                if b=='B8A':
                    channels.insert(8,ch)
                else:
                    channels.append(ch)
        img = np.dstack(channels)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, target = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, target

    def __len__(self):
        return len(self.indices)






if __name__ == '__main__':


    data_path = 'eurosat'
    batchsize = 4
    
    eurosat_dataset = EurosatDataset(root='eurosat')

    from sklearn.model_selection import train_test_split
    indices = np.arange(len(eurosat_dataset))
    train_indices, test_indices = train_test_split(indices, train_size=0.5,stratify=eurosat_dataset.targets)

    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(112),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor(),
            ])

    val_transforms = cvtransforms.Compose([
            cvtransforms.Resize(128),
            cvtransforms.CenterCrop(112),
            cvtransforms.ToTensor(),
            ])

    train_dataset = Subset(eurosat_dataset, train_indices, train_transforms)
    test_dataset = Subset(eurosat_dataset, test_indices, val_transforms)

    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=False,num_workers=2,pin_memory=False,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=batchsize,shuffle=False,num_workers=2,pin_memory=False,drop_last=True)

    print('train_len: %d val_len: %d' % (len(train_dataset),len(val_dataset)))


