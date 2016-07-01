import find_mxnet
import mxnet as mx
import logging
import argparse
import os
import train_model
# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(description='train an image classifer on imagenet')
parser.add_argument('--network', type=str, default='inception-bn',
                    choices = ['alexnet', 'vgg', 'googlenet', 'inception-bn', 'inception-bn-full', 'inception-v3', 'vgg16'],
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str,default='./rec_224/',
                    help='the input data directory')
parser.add_argument('--model-prefix', type=str,default='./model/checkpoint',
                    help='the prefix of the model to load/save')
parser.add_argument('--lr', type=float, default=.001,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--clip-gradient', type=float, default=5.,
                    help='clip min/max gradient to prevent extreme value')
parser.add_argument('--num-epochs', type=int, default=30,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--finetune-from', type=str, default='model/Inception_BN-0039',
                    help="finetune from model")
parser.add_argument('--finetune-lr-scale', type=float, default=10,
                    help="finetune layer lr_scale")
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--gpus', type=str, default="0",
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--num-examples', type=int, default=216,
                    help='the number of training examples')
parser.add_argument('--num-classes', type=int, default=10,
                    help='the number of classes')
parser.add_argument('--dataset', type=str, default='ft',
                    help='dataset')
parser.add_argument('--log-file', type=str,
		    help='the name of log file')
parser.add_argument('--log-dir', type=str, default="./tmp/",
                    help='directory of the log file')
parser.add_argument('--train-dataset', type=str,
                    help='train dataset name')
parser.add_argument('--val-dataset', type=str,
                    help="validation dataset name")
parser.add_argument('--data-shape', type=int, default=224,
                    help='set image\'s shape')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes, args.dataset)

# data
def get_iterator(args, kv):
    data_shape = (3, args.data_shape, args.data_shape)
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.train_dataset),
        mean_img = "mean.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        #max_aspect_ratio = 0.35,
        max_rotate_angle = 10, #random rotate 30 degrees
        #max_random_contrast = 0.25,
        #max_random_illumination = 0.3,
        shuffle     = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.val_dataset),
        rand_crop   = False,
        rand_mirror = False,
        mean_img = "mean.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

# train
train_model.fit(args, net, get_iterator)
