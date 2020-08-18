import os
import sys
import numpy as np
import argparse
import time
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.fs_loader import FewShotLoader, sampler
from roi_data_layer.simulation_loader import SimulationLoader
from roi_data_layer.ocid_loader import OCIDLoader

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.fsod_logger import FSODLogger

from model.faster_rcnn.fsod import FSOD
from model.faster_rcnn.share1 import Share1RCNN
from model.faster_rcnn.faster_rcnn import FasterRCNN
from model.faster_rcnn.multihead_attention import MultiheadAttentionRCNN
from model.faster_rcnn.reweight_fsod import ReweightFSOD


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=12, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=8, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=True,
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')                      
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=16, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true', default=True)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        action='store_true', default=False)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    # ablation
    parser.add_argument('--way', dest='way',
                        help='num of support way',
                        default=2, type=int)
    parser.add_argument('--shot', dest='shot',
                        help='num of support shot',
                        default=5, type=int)
    parser.add_argument('--pe', dest='pos_encoding',
                        help='positional encoding',
                        action='store_true', default=False)
    parser.add_argument('--nh', dest='n_head',
                        help='num of multihead',
                        default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(args)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == 'simulated':
        args.imdb_name = "simulated_train"
        args.imdbval_name = "simulated_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "ocid":
        args.imdb_name = "ocid_2020_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']


    elif args.dataset == "coco60_set1":
        args.imdb_name = "coco_60_set1"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "coco60_set2":
        args.imdb_name = "coco_60_set2"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "coco60_set3":
        args.imdb_name = "coco_60_set3"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "coco60_set4":
        args.imdb_name = "coco_60_set4"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "0712_cocoscale":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    else:
        raise Exception('dataset not defined')

    args.cfg_file = "cfgs/res50.yml"
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # make results determinable
    np.random.seed(cfg.RNG_SEED)
    random_seed = 1996
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if args.cuda:
        cfg.CUDA = True

    # prepare output dir
    output_dir = args.save_dir + "/train/checkpoints" 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare data
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    if args.dataset == 'simulated':
        dataset = SimulationLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                            imdb.num_classes, training=True, num_way=args.way, num_shot=args.shot)
    elif args.dataset == 'ocid':
        dataset = OCIDLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                            imdb.num_classes, training=True, num_way=args.way, num_shot=args.shot)
    else:
        dataset = FewShotLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                            imdb.num_classes, training=True, num_way=args.way, num_shot=args.shot)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))
    sampler_batch = sampler(train_size, args.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holders
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    fs_gt_boxes = torch.FloatTensor(1)
    support_imgs = torch.FloatTensor(1)
    ori_gt_boxes = torch.FloatTensor(1)
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        fs_gt_boxes = fs_gt_boxes.cuda()
        support_imgs = support_imgs.cuda()
        ori_gt_boxes = ori_gt_boxes.cuda()
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    fs_gt_boxes = Variable(fs_gt_boxes)
    support_imgs = Variable(support_imgs)
    ori_gt_boxes = Variable(ori_gt_boxes)

    # initilize the network
    if args.net == 'fsod':
        fasterRCNN = FSOD(imdb.classes, pretrained=True, num_way=args.way, num_shot=args.shot)
    elif args.net == 'fr':
        fasterRCNN = FasterRCNN(imdb.classes, pretrained=True)
    elif args.net == 'share':
        fasterRCNN = Share1RCNN(imdb.classes, pretrained=True, num_way=args.way, num_shot=args.shot)
    elif args.net == 'multi':
        fasterRCNN = MultiheadAttentionRCNN(imdb.classes, args.n_head, pretrained=True, num_way=args.way, num_shot=args.shot)
    elif args.net == 'reweight':
        fasterRCNN = ReweightFSOD(imdb.classes, pretrained=True, num_way=args.way, num_shot=args.shot)
    else:
        raise Exception(f"network {args.net} is not defined")

    fasterRCNN.create_architecture()
    if args.cuda:
        fasterRCNN.cuda()

    # optimizer
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # load checkpoints
    load_dir = args.load_dir + "/train/checkpoints" 
    if args.resume:
        load_name = os.path.join(load_dir,
            'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    # initialize logger
    logger_save_dir = args.save_dir + "/train"
    tb_logger = FSODLogger(logger_save_dir)

    iters_per_epoch = int(train_size / args.batch_size)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in tqdm(range(iters_per_epoch)):
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                fs_gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
                support_imgs.resize_(data[4].size()).copy_(data[4])
                ori_gt_boxes.resize_(data[5].size()).copy_(data[5])

            fasterRCNN.zero_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, fs_gt_boxes, num_boxes, support_imgs, ori_gt_boxes)

            # loss and loss.mean() are the same here
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                        % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                            % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                info = {
                'loss': loss_temp,
                'loss_rpn_cls': loss_rpn_cls,
                'loss_rpn_box': loss_rpn_box,
                'loss_rcnn_cls': loss_rcnn_cls,
                'loss_rcnn_box': loss_rcnn_box
                }
                loss_temp = 0
                start = time.time()
                
        tb_logger.write(epoch, info, im_data, support_imgs, fs_gt_boxes)

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))


