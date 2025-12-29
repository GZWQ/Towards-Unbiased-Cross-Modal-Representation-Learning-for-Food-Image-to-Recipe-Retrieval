import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label-Retrieval Training')

    parser.add_argument('--common_path', help='common_path', default='/common/home/users/q/qingwang.2020/codes/')
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/common/foodaiDrive/Recipe/recipe/images')
    parser.add_argument('--output', metavar='DIR', default='./checkpoints')
    parser.add_argument('--lang', default='us')

    parser.add_argument('--id2label_path', type=str, default=None)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--process_weight', type=float, default=1.0)
    parser.add_argument('--retrieval_weight', type=float, default=1.0)
    parser.add_argument('--aug_type', type=str, default='ret')
    parser.add_argument('--gt_level', type=float, default=1.0)
    parser.add_argument('--approx_type', type=str, default='img2text')

    #### Retrieval

    parser.add_argument('--output_size', type=int, default=1024,
                        help='Dimensionality of the output embeddings.')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Value of the margin for the triplet loss.')
    parser.add_argument('--recipe_loss_weight', type=float, default=1.0,
                        help='Weight value for the loss computed on recipe-only samples.')
    parser.add_argument('--dic_type', type=str, default='textual')

    parser.add_argument('--dataname', help='dataname', default='recipe1M', choices=['recipe1M'])
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')

    parser.add_argument('--num_class', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=True,
                        help='disable_torch_grad_focal_loss in asl')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                        help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')

    # distribution training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3718', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=1234, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # data aug
    parser.add_argument('--cutout', action='store_true', default=True,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ')

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true',
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true',
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * raining
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()

    if 'swin' in args.backbone:
        args.hidden_dim = 1024

    # args.resume = os.path.join(args.output, args.dataname,args.backbone, 'model_best.pth.tar')
    return args
