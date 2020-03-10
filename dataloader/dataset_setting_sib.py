import numpy as np
import torchvision.transforms as transforms

def dataset_setting(dataset, nSupport):
    if dataset == 'MiniImageNet':
        mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean, std=std)
        trainTransform = transforms.Compose([transforms.RandomCrop(80, padding=8),
                                             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                             transforms.RandomHorizontalFlip(),
                                             lambda x: np.asarray(x),
                                             transforms.ToTensor(),
                                             normalize
                                            ])

        valTransform = transforms.Compose([transforms.CenterCrop(80),
                                            lambda x: np.asarray(x),
                                            transforms.ToTensor(),
                                            normalize])

        inputW, inputH, nbCls = 80, 80, 64

        trainDir = './data/Mini-ImageNet/train/'
        valDir = './data/Mini-ImageNet/val/'
        testDir = './data/Mini-ImageNet/test/'
        episodeJson = './data/Mini-ImageNet/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
                else './data/Mini-ImageNet/val1000Episode_5_way_5_shot.json'

    else:
        raise ValueError('Do not support other datasets yet.')

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
