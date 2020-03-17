#   Copyright (c) 2020 Yaoyao Liu. All Rights Reserved.
#   Some files of this repository are modified from https://github.com/hushell/sib_meta_learn
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import os.path as osp
import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
TRAIN_PATH = osp.join(ROOT_PATH, 'data/tiered/train')
VAL_PATH = osp.join(ROOT_PATH, 'data/tiered/val')
TEST_PATH = osp.join(ROOT_PATH, 'data/tiered/test')

class TieredImageNet(Dataset):
    """ Usage: 
    """
    def __init__(self, setname, args, train_aug=False):
        if setname=='train':
            THE_PATH = TRAIN_PATH
        elif setname=='test':
            THE_PATH = TEST_PATH
        elif setname=='val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong setname.')            
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if os.path.isdir(osp.join(THE_PATH, label))]

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if train_aug:
            image_size = 80
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

