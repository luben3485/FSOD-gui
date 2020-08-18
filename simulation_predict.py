import numpy as np
import random
import time
import os
import cv2
import torch
import copy
from pathlib import Path
from PIL import Image
from scipy.misc import imread, imsave
from torch.autograd import Variable

from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import prep_im_for_blob, im_list_to_blob
from model.roi_layers import nms
from model.faster_rcnn.resnet import resnet
from model.utils.net_utils import vis_detections


def get_model(load_path, n_shot, cuda=True):
    fasterRCNN = resnet(['bg', 'fg'], 50, pretrained=False, num_way=2, num_shot=n_shot)
    fasterRCNN.create_architecture()
    print("load checkpoint %s" % (load_path))
    checkpoint = torch.load(load_path)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    fasterRCNN.eval()
    if cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()
    return fasterRCNN

def prepare_variable():
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    support_ims = torch.FloatTensor(1)

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

    return im_data, im_info, num_boxes, gt_boxes, support_ims


if __name__ == '__main__':


    SUPPORT_SIZE = 320
    POOL = ['85.jpg', '44.jpg', '61.jpg', '73.jpg', '94.jpg']
    CWD = os.getcwd()

    for TEST_SHOT in [1, 3, 5]:
        model_dir = os.path.join(CWD, 'models/simu_finetuned/train/checkpoints')
        load_path = os.path.join(model_dir,
                    'faster_rcnn_{}_{}_{}.pth'.format(1, 27, 1248))
        model = get_model(load_path, TEST_SHOT)

        query_name = '002468.jpg'
        file_path = os.path.join(CWD, 'data/simulated/images/simulated2020/' + query_name)
        im_data = imread(file_path)[:,:,::-1].copy()  # rgb -> bgr
        im2show = copy.deepcopy(im_data)
        target_size = cfg.TRAIN.SCALES[0]
        im_data, im_scale = prep_im_for_blob(im_data, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        im_data = torch.from_numpy(im_data)
        info = np.array([[im_data.shape[0], im_data.shape[1], im_scale]], dtype=np.float32)
        info = torch.from_numpy(info)
        dets = torch.from_numpy(np.array([0]))
        n_boxes = torch.from_numpy(np.array([0]))
        query = im_data.permute(2, 0, 1).contiguous().unsqueeze(0)

        support_data_all = np.zeros((TEST_SHOT, 3, SUPPORT_SIZE, SUPPORT_SIZE), dtype=np.float32)
        for i in range(TEST_SHOT):
            file_name = POOL[i]
            _path = os.path.join(CWD, 'data/simulated/supports/' + file_name)
            support_im = imread(_path)[:,:,::-1]  # rgb -> bgr
            target_size = np.min(support_im.shape[0:2])  # don't change the size
            support_im, _ = prep_im_for_blob(support_im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
            _h, _w = support_im.shape[0], support_im.shape[1]
            if _h > _w:
                resize_scale = float(SUPPORT_SIZE) / float(_h)
                unfit_size = int(_w * resize_scale)
                support_im = cv2.resize(support_im, (unfit_size, SUPPORT_SIZE), interpolation=cv2.INTER_LINEAR)
            else:
                resize_scale = float(SUPPORT_SIZE) / float(_w)
                unfit_size = int(_h * resize_scale)
                support_im = cv2.resize(support_im, (SUPPORT_SIZE, unfit_size), interpolation=cv2.INTER_LINEAR)
            h, w = support_im.shape[0], support_im.shape[1]
            support_data_all[i, :, :h, :w] = np.transpose(support_im, (2, 0, 1))
        supports = torch.from_numpy(support_data_all).unsqueeze(0)

    

        im_data, im_info, num_boxes, gt_boxes, support_ims = prepare_variable()
        data = [query, info, dets, n_boxes, supports]
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            support_ims.resize_(data[4].size()).copy_(data[4])

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = model(im_data, im_info, gt_boxes, num_boxes, support_ims)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        box_deltas = bbox_pred.data

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        # re-scale boxes to the origin img scale
        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        thresh = 0.05
        inds = torch.nonzero(scores[:,1]>thresh).view(-1)
        cls_scores = scores[:,1][inds]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = pred_boxes[inds, :]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]

        im2show = vis_detections(im2show, ' ', cls_dets.cpu().numpy(), 0.5)

        # im_pred = vis_bbox(im, cls_dets.cpu().numpy())
        output_path = os.path.join(CWD, 'data/simulated/output/tmp_' + str(TEST_SHOT) + '.jpg')
        cv2.imwrite(output_path, im2show)
        # print(cls_dets.size())
        print(cls_dets)


