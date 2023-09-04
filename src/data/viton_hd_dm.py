import os.path

from lightning import LightningDataModule
import torch
import numpy as np
import torchvision.transforms
from PIL import Image
import cv2
import random
from pathlib import Path
from torchvision import transforms


class ds(torch.utils.data.Dataset):
    def __init__(self, root : str, train_or_test : str, paired_unpaired : str, train_transforms : torchvision.transforms.Compose =None, test_transforms : torchvision.transforms.Compose=None)-> None:
        super().__init__()
        self.root = Path(root)
        self.train_or_test = train_or_test
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.paired_unpaired = paired_unpaired

        self.dataset_folders = ['agnostic-v3.2',
                                'cloth',
                                'cloth-mask',
                                'image',
                                'image-densepose',
                                'image-parse-agnostic-v3.2',
                                'image-parse-v3',
                                'openpose_img',
                                'openpose_json']

        if train_or_test == 'train':
            self.records = Path(self.root).joinpath('train_pairs.txt').read_text().split('\n')
            print(f"found {len(self.records)} in the dataset")
        else:
            self.records = Path(self.root).joinpath('test_pairs.txt').read_text().split('\n')

    def __len__(self)->int:
        return len(self.records)

    def __getitem__(self, item)->dict:
        single_record = self.records[item]
        if self.paired_unpaired == 'unpaired':
            # if len(single_record.split(' ')) == 2:
            target_person, cloth = single_record.split(' ')
            target_person = self.root.joinpath('train', self.dataset_folders[3], target_person)
            cloth = self.root.joinpath('train', self.dataset_folders[1], cloth)
            mask_cloth = self.root.joinpath('train', self.dataset_folders[2], cloth)
            # TODO : add code for paired images reading
        elif self.paired_unpaired == 'paired':
            # select the left and the right images, randomly say left is the target person and right is the cloth or vice versa
            records_ = single_record.split(' ')
            target_person = random.choice(records_)
            records_.remove(target_person)
            cloth = records_[0]

            target_person = self.root.joinpath('train', self.dataset_folders[3], target_person)
            cloth = self.root.joinpath('train', self.dataset_folders[1], cloth)
            mask_cloth = self.root.joinpath('train', self.dataset_folders[2], cloth.name)
            # mask_cloth = self.root.joinpath('train', "cloth-mask", cloth.name)
            Path(mask_cloth).exists()

        # ============================================================
        #                         transformations
        # ============================================================

        # read the images
        target_person = Image.open(target_person)
        cloth = Image.open(cloth)
        # read the masked cloth image as binary image, it will be the label
        # print(f"reading mask from {str(mask_cloth)}")
        # print(f"reading target person from {target_person.size}")
        # print(f"reading cloth from {cloth.size}")
        mask_cloth = cv2.imread(str(mask_cloth), cv2.IMREAD_GRAYSCALE)

        # if training phase and train transforms are not None
        if self.train_or_test == 'train' and self.train_transforms:
            target_person = self.train_transforms(target_person)
            cloth = self.train_transforms(cloth)
            # for the cloth, read the image and then apply binarization to get the mask
            mask_cloth = np.array(mask_cloth)  # dtype = uint8



            ret, thresh = cv2.threshold(mask_cloth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_cloth = transforms.ToTensor()(thresh)  # dtype = float32, unique values = 0, 1, shape = (1, H, W)
        elif self.train_or_test == 'test' and self.test_transforms:
            target_person = self.test_transforms(target_person)
            cloth = self.test_transforms(cloth)
            # for the cloth, read the image and then apply binarization to get the mask
            mask_cloth = np.array(mask_cloth)  # dtype = uint8
            ret, thresh = cv2.threshold(mask_cloth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_cloth = transforms.ToTensor()(thresh)  # dtype = float32, unique values = 0, 1

        return {'target_person': target_person, 'cloth': cloth, 'mask_cloth': mask_cloth}




class ds_module(LightningDataModule):
    def __init__(self,
                            root :str,
                            stage : str,
                            paired_unpaired :str,
                            bs :int,
                            workers :int

                 ):

        super().__init__()
        self.save_hyperparameters()

        # ===============================================================
        #                Defining the transformations
        # ===============================================================

        self.train_trf = transforms.Compose(
            [
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) # range [0.0, 1.0] -> [-1.0, 1.0]
            ]
        )

        self.test_trf = transforms.Compose(
            [
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) # range [0.0, 1.0] -> [-1.0, 1.0]
            ]
        )

    def setup(self, stage: str) -> None:
        self.train_ds = ds(
                            self.hparams.root,
                            self.hparams.stage,
                            self.hparams.paired_unpaired,
                            train_transforms=self.train_trf,
                            test_transforms=self.test_trf

                        )

    def train_dataloader(self)-> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.hparams.bs, shuffle=True, num_workers=self.hparams.workers)


if __name__ == '__main__':
    dm = ds_module(
        root = "/mnt/data1/sabbih/repos/clothes_transfer/LLM_finetune/pl_lightning_controlnet/pl-lightning-controlnet_small/viton_hd",
    stage = 'train',
    paired_unpaired = 'paired',
    bs= 12,
    workers = 2

    )
    dm.setup('train')
    dl = dm.train_dataloader()

    batch = next(iter(dl))

