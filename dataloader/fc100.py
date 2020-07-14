import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from os.path import expanduser

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args=None):
        data_base_dir = 'data/FC100'
        if os.path.exists(data_base_dir):
            pass
        else:
            print ('Download FC100 from Google Drive.')
            os.makedirs(data_base_dir)
            os.system('sh scripts/download_fc100.sh')

        TRAIN_PATH = 'data/FC100/train'
        VAL_PATH = 'data/FC100/val'
        TEST_PATH = 'data/FC100/test'
      
        if setname=='train':
            THE_PATH = TRAIN_PATH
            label_list = os.listdir(THE_PATH)
        elif setname=='test':
            THE_PATH = TEST_PATH
            label_list = os.listdir(THE_PATH)
        elif setname=='val':
            THE_PATH = VAL_PATH
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.') 
          
        data = []
        label = []

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92,92]),
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
