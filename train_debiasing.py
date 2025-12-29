import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from eval import computeAverageMetrics, pkl_dump
from eval_saving import computeAverageMetrics as computeAverageMetrics_saving

from config import parser_args

import _init_paths
from dataset.get_dataset import get_datasets, collate_fn
# from dataset.get_dataset_mmres import get_datasets, collate_fn

from utils.logger import setup_logger
import models
import models.aslloss
import models.tripleloss
from models.query2label_retrieval import build_q2l
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict

from collections import defaultdict

from utils import utils

import warnings
warnings.filterwarnings("ignore")


def get_args():
    args = parser_args()

    size_aug = 'Size{}_Aug{}'.format(args.img_size, args.aug_type)
    args.pretrained_path = './checkpoints/{}/resnet50/{}/Retrieval/'.format(args.dataname, size_aug)
    suf = 'End2End_Ret{}'.format(args.cls_weight)

    args.output = os.path.join(args.output, args.dataname, args.backbone, size_aug, 'Debiasing', suf)
    os.makedirs(args.output, exist_ok=True)
    return args, suf

best_recall1 = 0

def main():
    args,suf = get_args()

    id2labels = utils.load_pkl('/common/home/users/q/qingwang.2020/codes/HT_Debiasing/data/recipe1M/id2labels_train.pkl')
    ingredients_cls = []
    for k, v in id2labels.items():
        ingredients_cls.extend(v)
    args.num_class = len(set(ingredients_cls))

    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True

    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name=suf)
    logger.info("Command: " + ' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    print(args)

    return main_worker(args, logger)

def main_worker(args, logger):
    global best_recall1

    # Data loading code
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    args.vocab_size = len(train_dataset.get_vocab())

    # build model
    model = build_q2l(args, model_type='Qeruy2Label_DebiasingHard')
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # criterion
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )
    ranking_loss = models.tripleloss.TripletLoss(margin=args.margin)

    # optimizer
    args.lr_mult = args.batch_size / 256
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else:
        raise NotImplementedError

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    # optionally resume from a checkpoint
    if args.evaluate:
        args.resume = os.path.join(args.output, 'model_best.pth.tar')
    else:
        args.resume = os.path.join(args.pretrained_path, 'model_best.pth.tar')
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            # import ipdb; ipdb.set_trace()
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            # model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

        if not args.evaluate:

            print('Checking pretrained model performance:')

            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
                num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=test_sampler)

            # validate(test_loader, test_dataset, model, criterion, args, logger, split='test')

    # Data loading code
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=test_sampler)

    if args.evaluate:
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        #     num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, sampler=test_sampler)

        validate(test_loader, test_dataset, model, criterion, args, logger, split='test')

        return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)

    pres = AverageMeter('precision', ':5.5f', val_only=True)
    recs = AverageMeter('recall', ':5.5f', val_only=True)
    f1s = AverageMeter('f1 score', ':5.5f', val_only=True)

    medR = AverageMeter('medR', ':5.5f', val_only=True)
    r1s = AverageMeter('recall1', ':5.5f', val_only=True)
    r5s = AverageMeter('recall5', ':5.5f', val_only=True)
    r10s = AverageMeter('recall10', ':5.5f', val_only=True)

    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, pres, recs, f1s, medR, r1s, r5s, r10s],
        prefix='=> Val Epoch: ')

    scheduler = utils.get_scheduler(optimizer)

    end = time.time()
    best_epoch = -1

    precision_list = []
    recall_list = []
    f1_list = []

    medr_list = []
    r1_list = []
    r5_list = []
    r10_list = []

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(train_loader, model, criterion, ranking_loss, optimizer, epoch, args, logger)
        scheduler.step()

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:

            # evaluate on validation set
            loss, prf, recalls = validate(val_loader, val_dataset, model, criterion, args, logger)
            losses.update(loss)

            pres.update(prf['p_clswise'])
            recs.update(prf['r_clswise'])
            f1s.update(prf['f1_clswise'])

            medR.update(recalls['medr'])
            r1s.update(recalls['recall_1'])
            r5s.update(recalls['recall_5'])
            r10s.update(recalls['recall_10'])

            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            precision_list.append(prf['p_clswise'])
            recall_list.append(prf['r_clswise'])
            f1_list.append(prf['f1_clswise'])

            medr_list.append(recalls['medr'])
            r1_list.append(recalls['recall_1'])
            r5_list.append(recalls['recall_5'])
            r10_list.append(recalls['recall_10'])

            progress.display(epoch, logger)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_loss', loss, epoch)

            state_dict = model.state_dict()
            is_best = recalls['recall_1']  > best_recall1
            if is_best:
                best_epoch = epoch
            best_recall1 = max(best_recall1, recalls['recall_1'])
            logger.info("{} | Set best Recall1 {} in ep {}".format(epoch, best_recall1, best_epoch))

            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.backbone,
                    'state_dict': state_dict,
                    'best_recall1': best_recall1,
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))

    print("Best Recall1:", best_recall1)
    args.evaluate = True
    args.resume = os.path.join(args.output, 'model_best.pth.tar')
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

        if 'state_dict' in checkpoint:
            state_dict = clean_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            state_dict = clean_state_dict(checkpoint['model'])
        else:
            raise ValueError("No model or state_dicr Found!!!")
        logger.info("Omitting {}".format(args.resume_omit))
        # import ipdb; ipdb.set_trace()
        for omit_name in args.resume_omit:
            del state_dict[omit_name]
        model.module.load_state_dict(state_dict, strict=False)
        # model.module.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        del checkpoint
        del state_dict
        torch.cuda.empty_cache()
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
    validate(test_loader, test_dataset, model, criterion, args, logger, split='test')

    if summary_writer:
        summary_writer.close()

    return 0

def train(train_loader, model, criterion, ranking_loss, optimizer, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    losses = AverageMeter('Loss', ':5.3f')
    losses_cls = AverageMeter('Loss_CLS', ':.1f')
    losses_retrieval = AverageMeter('Loss_RET', ':.3f')
    losses_recipe = AverageMeter('Loss_REC', ':.3f')

    lr = AverageMeter('LR', ':.3e', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [lr, losses, losses_cls, losses_retrieval, losses_recipe],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    for i, (images, target, img_idx, title, ingrs, instrs) in enumerate(train_loader):
        # measure data loading time

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        loss_recipe = 0
        # from timeit import default_timer as timer
        with torch.cuda.amp.autocast(enabled=args.amp):
            # start = timer()
            output, img_feat, recipe_feat, proj_feats, _ = model(images, title, ingrs, instrs)
            # end = timer()
            # print(end-start)
            # exit(0)
            loss_cls = criterion(output, target)

            loss_ranking = ranking_loss(img_feat, recipe_feat)
            if args.recipe_loss_weight > 0:
                # compute triplet loss on pairs of raw and projected
                # feature vectors for all recipe component pairs

                # count number of component pairs for averaging
                c = 0
                names = ['title', 'ingredients', 'instructions']
                # for every recipe component
                for raw_name in names:
                    # get the original feature (not projected) as the query (q)
                    q = proj_feats['raw'][raw_name]
                    # for every other recipe component (proj_name)
                    for proj_name in names:
                        if proj_name != raw_name:
                            # get the projection from its raw feature
                            # to the query recipe component as value (v)
                            # (e.g. query=title, value=proj_ingredient2title(ingredient))
                            v = proj_feats[proj_name][raw_name]
                            loss_recipe += ranking_loss(q, v)
                            c += 1
                loss_recipe /= c

            loss_cls = loss_cls*args.cls_weight
            loss_ranking = loss_ranking * args.retrieval_weight
            loss_recipe = loss_recipe * args.retrieval_weight

            loss = loss_cls + loss_ranking + loss_recipe

            if args.loss_dev > 0:
                loss *= args.loss_dev

        # record loss
        losses.update(loss.item(), images.size(0))
        losses_cls.update(loss_cls.item(), images.size(0))
        losses_retrieval.update(loss_ranking.item(), images.size(0))
        losses_recipe.update(loss_recipe.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr.update(get_learning_rate(optimizer))

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg

def get_metrics_mmres(args, img_feats, recipe_feats, ids):

    if args.evaluate:
        rank_size_list = [100, 200, 500, len(img_feats)]
        repeat_num_list = [5, 4, 1, 1]
    else:
        rank_size_list = [len(img_feats)]
        repeat_num_list = [1]

    print('rank_size_list',rank_size_list)

    for rank_size, repeat_num in zip(rank_size_list, repeat_num_list):

        metrics_epoch = defaultdict(list)
        retrieval_metrics = computeAverageMetrics(img_feats, recipe_feats,
                                                  rank_size, repeat_num, forceorder=True)
        # retrieval_metrics = computeAverageMetrics(recipe_feats, img_feats,
        #                                           rank_size, 10, forceorder=len(rank_size_list) == 1)
        for k, v in retrieval_metrics.items():
            metrics_epoch[k] = v
        avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}

        text_ = "Retrieval of {}".format(rank_size)
        values = []
        for k, v in avg_metrics.items():
            text_ += ", " + k + ": {:.4f}"
            values.append(v)
        str_ = text_.format(*values)
        print(str_)

    return avg_metrics

def get_metrics(args, img_feats, recipe_feats, ids):

    if args.evaluate:
        rank_size_list = [1000, 10000, 20000, 30000, 40000, 50000]
    else:
        rank_size_list = [1000]

    print('rank_size_list',rank_size_list)

    for rank_size in rank_size_list:

        metrics_epoch = defaultdict(list)
        # retrieval_metrics = computeAverageMetrics(img_feats, recipe_feats,
        #                                           rank_size, 10, forceorder=len(rank_size_list)==1)
        retrieval_metrics = computeAverageMetrics(recipe_feats, img_feats,
                                                  rank_size, 10, forceorder=len(rank_size_list) == 1)
        for k, v in retrieval_metrics.items():
            metrics_epoch[k] = v
        avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}

        text_ = "Retrieval of {}".format(rank_size)
        values = []
        for k, v in avg_metrics.items():
            text_ += ", " + k + ": {:.4f}"
            values.append(v)
        str_ = text_.format(*values)
        print(str_)

    return avg_metrics

def get_metrics_analysis(args, img_feats, recipe_feats, ids):

    rank_size_list = [len(img_feats)]

    print('rank_size_list',rank_size_list, len(recipe_feats))

    for rank_size in rank_size_list:

        metrics_epoch = defaultdict(list)
        retrieval_metrics, id2results = computeAverageMetrics_saving(img_feats, recipe_feats,ids,rank_size)
        # retrieval_metrics, id2results = computeAverageMetrics_saving(recipe_feats,img_feats,ids,rank_size)

        for k, v in retrieval_metrics.items():
            metrics_epoch[k] = v
        avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}

        text_ = "Retrieval of {}".format(rank_size)
        values = []
        for k, v in avg_metrics.items():
            text_ += ", " + k + ": {:.4f}"
            values.append(v)
        str_ = text_.format(*values)
        print(str_)

        print('Saving id2results.pkl to', args.output)
        utils.dump_pkl(os.path.join(args.output, 'id2results_i2r.pkl'), id2results)

    return avg_metrics

def eval_F1(args, results, gt_labels, ids=None, split='val'):
    y_pred = np.asarray(results) > 0
    macro_f1 = f1_score(gt_labels, y_pred, average='macro')
    p_clswise = precision_score(gt_labels, y_pred, average='macro')
    r_clswise = recall_score(gt_labels, y_pred, average='macro')

    p_all = precision_score(gt_labels, y_pred, average='micro')
    r_all = recall_score(gt_labels, y_pred, average='micro')


    return macro_f1, {"p_clswise": p_clswise, "r_clswise": r_clswise, 'f1_clswise': 2 * p_clswise * r_clswise / (p_clswise + r_clswise),
                      'p_all':p_all, 'r_all':r_all,'f1_all': 2*p_all*r_all/(p_all+r_all)
                      }

@torch.no_grad()
def validate(val_loader,test_dataset, model, criterion, args, logger, split='val'):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    img_feats_total, recipe_feats_total = None, None
    img_feats_pure, img_feats_z = None, None
    img_feats, labels, allids = None, None, []
    path2ifeat = {}
    path2ifeat_text = {}
    with torch.no_grad():
        end = time.time()
        for i, (images, target, ids, title, ingrs, instrs) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output, img_feat, recipe_feat, _, intermediate_feats = model(images, title, ingrs, instrs, prob_ing_r=target)

                loss = criterion(output, target)
                if args.loss_dev > 0:
                    loss *= args.loss_dev

            # record loss
            losses.update(loss.item(), images.size(0))
            img_feat_npy = img_feat.cpu().detach().numpy()
            for one_path, one_feat in zip(ids, img_feat_npy):
                path2ifeat[one_path] = one_feat
            recipe_feat_npy = recipe_feat.cpu().detach().numpy()
            for one_path, one_feat in zip(ids, recipe_feat_npy):
                path2ifeat_text[one_path] = one_feat

            if img_feats is not None:
                img_feats = np.vstack((img_feats, output.cpu().detach().numpy()))
                labels = np.vstack((labels, target.cpu().detach().numpy()))

                img_feats_total = np.vstack((img_feats_total, img_feat.cpu().detach().numpy()))
                recipe_feats_total = np.vstack((recipe_feats_total, recipe_feat.cpu().detach().numpy()))

                img_feats_pure = np.vstack((img_feats_pure, intermediate_feats['img_feats_pure'].cpu().detach().numpy()))
                img_feats_z = np.vstack((img_feats_z, intermediate_feats['img_feats_z'].cpu().detach().numpy()))

            else:
                img_feats = output.cpu().detach().numpy()
                labels = target.cpu().detach().numpy()

                img_feats_total = img_feat.cpu().detach().numpy()
                recipe_feats_total = recipe_feat.cpu().detach().numpy()

                img_feats_pure = intermediate_feats['img_feats_pure'].cpu().detach().numpy()
                img_feats_z = intermediate_feats['img_feats_z'].cpu().detach().numpy()

            allids.extend(ids)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        logger.info('=> synchronize...')
        pkl_dump(os.path.join(args.output, 'path2ifeat.pkl'), path2ifeat)
        pkl_dump(os.path.join(args.output, 'path2ifeat_text.pkl'), path2ifeat_text)

        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )

        if dist.get_world_size() > 1:
            dist.barrier()
        f1_perf, prf = eval_F1(args, img_feats, labels, allids, split)
        text_ = "CLS. "
        values = []
        for k, v in prf.items():
            print(k, v)
            text_ += ", " + k + ": {:.4f}"
            values.append(v)
        str_ = text_.format(*values)
        print(str_)

        allids = np.array(allids)

        avg_metrics = get_metrics(args, img_feats_total, recipe_feats_total, allids)
        text_ = "Retrieval: "
        values = []
        for k, v in avg_metrics.items():
            text_ += ", " + k + ": {:.4f}"
            values.append(v)
        str_ = text_.format(*values)
        print(str_)

    return loss_avg, prf, avg_metrics


##################################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist


if __name__ == '__main__':
    # python train_debiasing.py --dist-url tcp://127.0.0.1:4503 --cls_weight 1.0 --retrieval_weight 1.0 --batch-size 64 --dataname semantic_food_500 --backbone resnet50
    main()
