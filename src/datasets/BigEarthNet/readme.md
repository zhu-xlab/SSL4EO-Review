
## BigEarthNet data preparation: SeCo LMDB 

update: RGB only  -->  multi-band processing
* remove `Image.fromarray()`
* use `cvtorchvision.cvtransforms` instead of torchvision.transforms, run `pip install opencv-torchvision-transforms-yuzhiyang` to install the package

### 1. download data
[Option 1] download data before making dataset
* download bigearthnet from http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz and extract
* download csv file noting bad_patches:
  * http://bigearth.net/static/documents/patches_with_seasonal_snow.csv
  * http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv
* make train/val/test list of all image patch names, or download from:
  * https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt
  * https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt
  * https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt


[Option 2] download data when making dataset (some bugs here, to be checked)
```
python bigearthnet_dataset_seco.py --download True
```


### 2. create dataset
[Option 1] create normal dataset

```
from bigearthnet_dataset_seco import Bigearthnet
from cvtorchvision import cvtransforms

data_dir = '' # parent folder of BigEarthNet-v1 data
train_transforms = cvtransforms.Compose([cvtransforms.ToTensor()])

train_dataset = Bigearthnet(root=data_dir, split='train', bands=None, transform=train_transforms, target_transform=None, download=False, use_new_labels=True)

train_loader = DataLoader(train_dataset,
							batch_size=512,
							shuffle=True,
							num_workers=2,
							pin_memory=True,
							drop_last=True)	
```

[option 2] create lmdb dataset
* make lmdb dataset: 
	```
	python bigearthnet_dataset_seco.py --make_lmdb_dataset True --save_dir [your path to store lmdb file] --data_dir [your path to BigEarthNet data]
	```
* load lmdb dataset:

	```
	from cvtorchvision import cvtransforms
	from bigearthnet_dataset_seco_lmdb import LMDBDataset, random_subset
	
	
	data_dir = '' # parent folder of lmdb file
	
	train_transforms = cvtransforms.Compose([cvtransforms.ToTensor()])
	train_frac = 1
	seed = 42
    train_dataset = LMDBDataset(lmdb_file=os.path.join(data_dir, 'train.lmdb'), transform=train_transforms)
	
	if train_frac is not None and train_frac < 1:
		train_dataset = random_subset(train_dataset, train_frac, seed) 	
	

	train_loader = DataLoader(train_dataset,
							  batch_size=512,
							  shuffle=True,
							  num_workers=2,
							  pin_memory=True,
							  drop_last=True)
	```
