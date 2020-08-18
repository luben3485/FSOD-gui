import os
import sys
import numpy as np
import argparse
import time
import pickle
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.inference_loader import InferenceLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.fsod_logger import FSODInferenceLogger

from model.faster_rcnn.faster_rcnn import FasterRCNN


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set_cfg', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default='True',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true', default=True)
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    # ablation
    parser.add_argument('--epi', dest='num_episode',
                        help='num of episode to use',
                        default=5, type=int)
    parser.add_argument('--np', dest='not_pure',
                        help='whether select high quality supports',
                        action='store_true', default=False)
    parser.add_argument('--ms', dest='more_scale',
                        help='using coco anchor scale',
                        action='store_true', default=False)                     
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "set1":
        args.imdbval_name = "coco_set1_ep0"
        args.imdbval_general_name = "coco_set1_ep"
    elif args.dataset == "set2":
        args.imdbval_name = "coco_set2_ep0"
        args.imdbval_general_name = "coco_set2_ep"
    elif args.dataset == "set3":
        args.imdbval_name = "coco_set3_ep0"
        args.imdbval_general_name = "coco_set3_ep"
    elif args.dataset == "set4":
        args.imdbval_name = "coco_set4_ep0"
        args.imdbval_general_name = "coco_set4_ep"
    else:
        raise Exception("dataset is not defined")

    if args.more_scale:
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    else:
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/res50.yml"
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # prepare roidb
    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    cwd = os.getcwd()
    if args.not_pure:
        support_dir = os.path.join(cwd, 'data/supports', args.dataset + '_random')
    else:
        support_dir = os.path.join(cwd, 'data/supports', args.dataset)

    # load dir
    input_dir = os.path.join(args.load_dir, "train/checkpoints")
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network
    if args.net == 'fr':
        fasterRCNN = FasterRCNN(imdb.classes, pretrained=True)
    else:
        raise Exception("network is not defined")

    fasterRCNN.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    # initilize the tensor holder
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    support_ims = torch.FloatTensor(1)
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        support_ims = support_ims.cuda()
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    support_ims = Variable(support_ims)

    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()

    # prepare holder for predicted boxes
    start = time.time()
    max_per_image = 100
    thresh = 0.05
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                for _ in range(imdb.num_classes)]
    _t = {'im_detect': time.time(), 'misc': time.time()}

    # initialize logger
    if args.vis:
        logger_save_dir = args.load_dir + "/test"
        tb_logger = FSODInferenceLogger(logger_save_dir)
        logger_save_step = 0

    # inference
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
    num_episode = args.num_episode
    for n_ep in range(num_episode):
        args.imdbval_name = args.imdbval_general_name + str(n_ep)
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
        imdb.competition_mode(on=True)
        dataset = InferenceLoader(imdb, roidb, ratio_list, ratio_index, support_dir, 
                            1, 21, num_shot=1, training=False, normalize = False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0,
                                pin_memory=True)
        data_iter = iter(dataloader)
        print('{:d} roidb entries'.format(len(roidb)))

        for i in range(num_images):
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
                support_ims.resize_(data[4].size()).copy_(data[4])

            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, support_ims, gt_boxes)

            scores = cls_prob.data  # [1, 300, 21]
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            # re-scale boxes to the origin img scale
            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()  # [300, 21]
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()

            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)

            selected_class = gt_boxes[0, 0, 4]

            for j in range(1, imdb.num_classes):
                if j != selected_class:
                    all_boxes[j][i] = empty_array
                    continue
                inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            # prepare gt
            if args.vis:
                img_with_selected_gtbox = np.copy(im)
                gt_dets = gt_boxes[0].cpu().numpy()
                gt_dets /= data[1][0][2].item()
                gt_dets[:, 4] = 1.
                gt_class_name = imdb._classes[int(gt_boxes[0, 0, 4])]
                img_with_selected_gtbox = vis_detections(img_with_selected_gtbox, gt_class_name, gt_dets, 0.8)

                img_pred = torch.from_numpy(im2show[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float()
                img_gt = torch.from_numpy(img_with_selected_gtbox[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float()
                support = support_ims.squeeze()[0].permute(1, 2, 0).cpu().numpy()
                h, w = img_pred.size(2), img_pred.size(3)
                support = cv2.resize(support, (w, h), interpolation=cv2.INTER_LINEAR)
                support = torch.from_numpy(support).permute(2, 0, 1).unsqueeze(0).float()
                inv_idx = torch.arange(2, -1, -1)
                support = support[:, inv_idx, :, :]

                tb_logger.write(img_gt, support, img_pred)

        cwd = os.getcwd()
        output_dir = os.path.join(cwd, 'output', args.net, args.imdbval_general_name + str(n_ep))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))
