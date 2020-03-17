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

import os
import torch
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import json
from torchvision import transforms
from torchvision.datasets import ImageFolder

def PilLoaderRGB(imgPath) :
    return Image.open(imgPath).convert('RGB')


class EpisodeSampler():
    def __init__(self, imgDir, nClsEpisode, nSupport, nQuery, transform, useGPU, inputW, inputH):
        self.imgDir = imgDir
        self.clsList = os.listdir(imgDir)
        self.nClsEpisode = nClsEpisode
        self.nSupport = nSupport
        self.nQuery = nQuery
        self.transform = transform

        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.tensorSupport = floatType(nClsEpisode * nSupport, 3, inputW, inputH)
        self.labelSupport = intType(nClsEpisode * nSupport)
        self.tensorQuery = floatType(nClsEpisode * nQuery, 3, inputW, inputH)
        self.labelQuery = intType(nClsEpisode * nQuery)
        self.imgTensor = floatType(3, inputW, inputH)

    def getEpisode(self):
        for i in range(self.nClsEpisode) :
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i

        clsEpisode = np.random.choice(self.clsList, self.nClsEpisode, replace=False)
        for i, cls in enumerate(clsEpisode) :
            clsPath = os.path.join(self.imgDir, cls)
            imgList = os.listdir(clsPath)

            imgCls = np.random.choice(imgList, self.nQuery + self.nSupport, replace=False)

            for j in range(self.nSupport) :
                img = imgCls[j]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery) :
                img = imgCls[j + self.nSupport]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        permSupport = torch.randperm(self.nClsEpisode * self.nSupport)
        permQuery = torch.randperm(self.nClsEpisode * self.nQuery)

        return {'SupportTensor':self.tensorSupport[permSupport],
                'SupportLabel':self.labelSupport[permSupport],
                'QueryTensor':self.tensorQuery[permQuery],
                'QueryLabel':self.labelQuery[permQuery]
                }


class BatchSampler():
    def __init__(self, imgDir, nClsEpisode, nSupport, nQuery, transform, useGPU, inputW, inputH, batchSize):
        self.episodeSampler = EpisodeSampler(imgDir, nClsEpisode, nSupport, nQuery,
                                             transform, useGPU, inputW, inputH)

        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.tensorSupport = floatType(batchSize, nClsEpisode * nSupport, 3, inputW, inputH)
        self.labelSupport = intType(batchSize, nClsEpisode * nSupport)
        self.tensorQuery = floatType(batchSize, nClsEpisode * nQuery, 3, inputW, inputH)
        self.labelQuery = intType(batchSize, nClsEpisode * nQuery)

        self.batchSize = batchSize

    def getBatch(self):
        for i in range(self.batchSize) :
            episode = self.episodeSampler.getEpisode()
            self.tensorSupport[i] = episode['SupportTensor']
            self.labelSupport[i] = episode['SupportLabel']
            self.tensorQuery[i] = episode['QueryTensor']
            self.labelQuery[i] = episode['QueryLabel']

        return {'SupportTensor':self.tensorSupport,
                'SupportLabel':self.labelSupport,
                'QueryTensor':self.tensorQuery,
                'QueryLabel':self.labelQuery
                }


class ValImageFolder(data.Dataset):
    def __init__(self, episodeJson, imgDir, inputW, inputH, valTransform, useGPU):
        with open(episodeJson, 'r') as f :
            self.episodeInfo = json.load(f)

        self.imgDir = imgDir
        self.nEpisode = len(self.episodeInfo)
        self.nClsEpisode = len(self.episodeInfo[0]['Support'])
        self.nSupport = len(self.episodeInfo[0]['Support'][0])
        self.nQuery = len(self.episodeInfo[0]['Query'][0])
        self.transform = valTransform
        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.tensorSupport = floatType(self.nClsEpisode * self.nSupport, 3, inputW, inputH)
        self.labelSupport = intType(self.nClsEpisode * self.nSupport)
        self.tensorQuery = floatType(self.nClsEpisode * self.nQuery, 3, inputW, inputH)
        self.labelQuery = intType(self.nClsEpisode * self.nQuery)

        self.imgTensor = floatType(3, inputW, inputH)
        for i in range(self.nClsEpisode) :
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i


    def __getitem__(self, index):
        for i in range(self.nClsEpisode) :
            for j in range(self.nSupport) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Support'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Query'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        return {'SupportTensor':self.tensorSupport,
                'SupportLabel':self.labelSupport,
                'QueryTensor':self.tensorQuery,
                'QueryLabel':self.labelQuery
                }

    def __len__(self):
        return self.nEpisode


def ValLoader(episodeJson, imgDir, inputW, inputH, valTransform, useGPU) :
    dataloader = data.DataLoader(ValImageFolder(episodeJson, imgDir, inputW, inputH,
                                                valTransform, useGPU),
                                 shuffle=False)
    return dataloader


def TrainLoader(batchSize, imgDir, trainTransform) :
    dataloader = data.DataLoader(ImageFolder(imgDir, trainTransform),
                                 batch_size=batchSize, shuffle=True, drop_last=True)
    return dataloader

