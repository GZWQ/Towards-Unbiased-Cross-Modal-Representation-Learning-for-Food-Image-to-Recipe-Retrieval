# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""
Dataset and dataloader functions
"""

import os
import json
import random
random.seed(1234)
import pickle
import torch
from torch.utils.data import Dataset
from utils.utils import get_token_ids, list2Tensors, load_pkl
import numpy as np
from random import choice
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Recipe1M(Dataset):

    def __init__(self, args, transform=None, split='train',max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15,
                 text_only_data=False):

        common_path = '/common/home/users/q/qingwang.2020/codes/HT_Debiasing/data/'
        self.args = args
        # load vocabulary
        self.vocab_inv = pickle.load(open(common_path+'vocab.pkl', 'rb'))
        self.vocab = {}
        for k, v in self.vocab_inv.items():
            if type(v) != str:
                v = v[0]
            self.vocab[v] = k

        # suffix to load text only samples or paired samples
        suf = '_noimages' if text_only_data else ''
        data_path = os.path.join(common_path+'recipe1M', split + suf + '.pkl')
        print(split, data_path)
        self.data = pickle.load(open(data_path,'rb'))
        self.root = args.dataset_dir

        self.split = split
        self.transform = transform

        self.max_ingrs = max_ingrs
        self.max_instrs = max_instrs
        self.max_length_ingrs = max_length_ingrs
        self.max_length_instrs = max_length_instrs

        self.text_only_data = text_only_data

        self.num_classes = args.num_class

        id2label_path = common_path+'recipe1M/id2labels_{}.pkl'.format(split)

        print('Loading id2labels', id2label_path)
        self.id2labels = load_pkl(id2label_path)

        self.ids = list(self.data.keys())


    def getLabelVector(self,categories):
        one_hot_tensor = torch.zeros(self.num_classes)

        for idx in categories:
            one_hot_tensor[idx] = 1

        return one_hot_tensor

    def __getitem__(self, idx):

        entry = self.data[self.ids[idx]]

        if not self.text_only_data:
            # loading images
            if self.split == 'train':
                # if training, pick an image randomly
                img_name = entry['images'][0]

            else:
                # if test or val we pick the first image
                img_name = entry['images'][0]

            img_name = '/'.join(img_name[:4])+'/'+img_name
            path = os.path.join(self.root, self.split, img_name)
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = None

        title = entry['title']
        ingrs = entry['ingredients']
        instrs = entry['instructions']

        real_id = self.ids[idx].split('_')[0]
        label = self.getLabelVector(self.id2labels[real_id])

        title = torch.Tensor(get_token_ids(title, self.vocab)[:self.max_length_instrs])
        instrs = list2Tensors([get_token_ids(instr, self.vocab)[:self.max_length_instrs] for instr in instrs[:self.max_instrs]])
        ingrs = list2Tensors([get_token_ids(ingr, self.vocab)[:self.max_length_ingrs] for ingr in ingrs[:self.max_ingrs]])
        return img, label, self.ids[idx], title, ingrs, instrs

    def __len__(self):
        return len(self.ids)

    def get_vocab(self):
        try:
            return self.vocab_inv
        except:
            return None

