# --------------------------------------------------------
# Quert2Label
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math, pickle
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.recipe_encoder import RecipeTransformerEncoder
from utils.misc import clean_state_dict


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class Qeruy2Label(nn.Module):
    def __init__(self, args, backbone, transfomer, text_encoder, num_class):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.text_encoder = text_encoder
        self.num_class = num_class

        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        in_feats = backbone.num_channels
        self.img_fc = nn.Linear(in_feats, args.output_size)
        self.merger_recipe = nn.ModuleList()
        self.merger_recipe = nn.Linear(512 * (3), args.output_size)

        # projection layers for self supervised recipe loss
        self.projector_recipes = nn.ModuleDict()
        names = ['title', 'ingredients', 'instructions']
        for name in names:
            self.projector_recipes[name] = nn.ModuleDict()
            for name2 in names:
                if name2 != name:
                    self.projector_recipes[name][name2] = nn.Linear(512, 512)

    def forward(self, input, title, ingrs, instrs, prob_ing_r=None):

        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d
        out = self.fc(hs[-1])

        # Image features
        img_feat = src.view(src.size(0), src.size(1),src.size(2)*src.size(3))
        img_feat = torch.mean(img_feat, dim=-1)
        img_feat = nn.Tanh()(self.img_fc(img_feat))

        # Text features
        text_features = []
        projected_text_features = {'title': {},
                                   'ingredients': {},
                                   'instructions': {},
                                   'raw': {}}

        elems = {'title': title, 'ingredients': ingrs, 'instructions': instrs}

        names = list(elems.keys())
        ingredient_feats = []
        instruction_feats = []
        for name in names:
            # for each recipe component, extracts features and projects them to all other spaces
            input_source = elems[name]
            text_feature, first_layer_feature = self.text_encoder(input_source, name)
            text_features.append(text_feature)
            projected_text_features['raw'][name] = text_feature

            if name == 'ingredients':
                ingredient_feats = first_layer_feature
            elif name == 'instructions':
                instruction_feats = first_layer_feature

            for name2 in names:
                if name2 != name:
                    projected_text_features[name][name2] = self.projector_recipes[name][name2](text_feature)
        recipe_feat = self.merger_recipe(torch.cat(text_features, dim=1))
        recipe_feat = nn.Tanh()(recipe_feat)
        return out, img_feat, recipe_feat, projected_text_features, ingredient_feats, instruction_feats

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))

class Qeruy2Label_DebiasingHard(nn.Module):
    def __init__(self, args, backbone, transfomer, text_encoder, num_class):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.text_encoder = text_encoder
        self.num_class = num_class

        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        in_feats = backbone.num_channels
        self.img_fc = nn.Linear(in_feats, args.output_size)
        self.merger_recipe = nn.ModuleList()
        self.merger_recipe = nn.Linear(512 * (3), args.output_size)

        # projection layers for self supervised recipe loss
        self.projector_recipes = nn.ModuleDict()
        names = ['title', 'ingredients', 'instructions']
        for name in names:
            self.projector_recipes[name] = nn.ModuleDict()
            for name2 in names:
                if name2 != name:
                    self.projector_recipes[name][name2] = nn.Linear(512, 512)

        self.sigmoid_func = nn.Sigmoid()
        # Debiasing module

        dic_path = os.path.join(args.pretrained_path, 'ingredient_dict_textual.pkl')

        print('Loading ', dic_path)
        with open(dic_path, 'rb') as h:
            ingredient_dict = pickle.load(h)
        ingredient_centers = []
        for k in range(len(ingredient_dict)):
            ingredient_centers.append(np.mean(ingredient_dict[k], axis=0))
        ingredient_centers = np.vstack(ingredient_centers)
        print('ing_dic size', np.shape(ingredient_centers))
        self.ing_dic = torch.tensor(ingredient_centers, dtype=torch.float)

        self.Wv = nn.Linear(512, args.output_size)
        print("Extra parameters are, {}({})".format('Wv', 'Linear'))

    def debiasing_module(self, prob_ing_r):
        device = prob_ing_r.get_device()
        dic_z = self.ing_dic.to(device)
        z_hat = torch.mm(prob_ing_r, self.Wv(dic_z))
        return z_hat

    def forward(self, input, title, ingrs, instrs, prob_ing_r=None):

        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        # Image features
        img_feat = src.view(src.size(0), src.size(1),src.size(2)*src.size(3))
        img_feat = torch.mean(img_feat, dim=-1)
        img_feat_pure = nn.Tanh()(self.img_fc(img_feat))

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0]  # B,K,d
        out = self.fc(hs[-1])

        feat2debug = {}
        # Debiasing
        logit = self.sigmoid_func(out)
        prob_ing_r = torch.where(logit < 0.5, torch.tensor(0.0), logit)
        device = input.get_device()
        prob_ing_r = prob_ing_r.to(device)

        prob_ing_r = prob_ing_r / torch.sum(prob_ing_r, dim=1).unsqueeze(dim=1)
        prob_ing_r = torch.nan_to_num(prob_ing_r)
        z_hat_img = self.debiasing_module(prob_ing_r)
        img_feat = img_feat_pure + z_hat_img
        # img_feat = img_feat_pure

        feat2debug['img_feats_pure'] = img_feat_pure
        feat2debug['img_feats_z'] = z_hat_img

        # Text features
        text_features = []
        projected_text_features = {'title': {},
                                   'ingredients': {},
                                   'instructions': {},
                                   'raw': {}}

        elems = {'title': title, 'ingredients': ingrs, 'instructions': instrs}

        names = list(elems.keys())
        for name in names:
            # for each recipe component, extracts features and projects them to all other spaces
            input_source = elems[name]
            text_feature, first_layer_feature = self.text_encoder(input_source, name)
            text_features.append(text_feature)
            projected_text_features['raw'][name] = text_feature
            for name2 in names:
                if name2 != name:
                    projected_text_features[name][name2] = self.projector_recipes[name][name2](text_feature)
        recipe_feat = self.merger_recipe(torch.cat(text_features, dim=1))
        recipe_feat = nn.Tanh()(recipe_feat)

        return out, img_feat, recipe_feat, projected_text_features, feat2debug

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


def build_q2l(args, model_type='Qeruy2Label'):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    text_encoder = RecipeTransformerEncoder(args.vocab_size)

    print('Building model', model_type)

    if model_type == 'Qeruy2Label':
        model = Qeruy2Label(
            args=args,
            backbone=backbone,
            transfomer=transformer,
            text_encoder=text_encoder,
            num_class=args.num_class
        )
    elif model_type == 'Qeruy2Label_DebiasingHard':
        model = Qeruy2Label_DebiasingHard(
            args=args,
            backbone=backbone,
            transfomer=transformer,
            text_encoder=text_encoder,
            num_class=args.num_class
        )
    else:
        print('Not Implemented', model_type)
        exit(0)


    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")

    return model


