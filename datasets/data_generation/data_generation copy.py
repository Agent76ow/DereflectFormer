import os
from os.path import join

import torch.backends.cudnn as cudnn

import sirs_dataset as datasets
from image_folder import read_fns

if __name__ == '__main__':

    datadir = os.path.dirname(os.path.abspath(__file__))

    datadir_syn = join(datadir, 'synthetic\\reflection_layer')
    datadir_real = join(datadir, 'real')

    train_dataset = datasets.CEILDataset(
        datadir_syn, read_fns(join(datadir, 'VOC2012_224_train_png.txt')), size=99999, enable_transforms=True,
        low_sigma=2, high_sigma=5,
        low_gamma=1.3, high_gamma=1.3)
    
    # train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True, if_align=True)
    # train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

    # train_dataloader_fusion = datasets.DataLoader(
    #     train_dataset_fusion, batch_size=1, shuffle=not True,
    #     num_workers=0, pin_memory=True)
