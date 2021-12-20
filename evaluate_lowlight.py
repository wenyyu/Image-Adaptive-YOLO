#! /usr/bin/env python
# coding=utf-8


import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config_lowlight import cfg
from core.yolov3_lowlight import YOLOV3
from core.config_lowlight import args
import random
import time
exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))


if args.use_gpu == 0:
    gpu_id = '-1'
else:
    gpu_id = args.gpu_id
    gpu_list = list()
    gpu_ids = gpu_id.split(',')
    for i in range(len(gpu_ids)):
        gpu_list.append('/gpu:%d' % int(i))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL
        self.isp_flag = cfg.YOLO.ISP_FLAG

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')
            self.input_data_clean   = tf.placeholder(tf.float32, [None, None, None, 3], name='input_data')


        model = YOLOV3(self.input_data, self.trainable, self.input_data_clean)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.image_isped, self.isp_params = \
            model.pred_sbbox, model.pred_mbbox, model.pred_lbbox, model.image_isped,model.filter_params

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image, image_name):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox, image_isped, isp_param = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.image_isped, self.isp_params],
            feed_dict={
                self.input_data: image_data,  # image_data*np.exp(lowlight_param*np.log(2)),
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        if self.isp_flag:
            print('ISP params :  ', isp_param)
            image_isped = np.clip(image_isped[0, ...]*255, 0, 255)
            image_isped = utils.image_unpreporcess(image_isped, [org_h, org_w])
        else:
            image_isped = np.clip(image, 0, 255)
            # image_isped = utils.image_unpreporcess(image_isped, [org_h, org_w])
            # cv2.imwrite(self.write_image_path + 'low'+ image_name, image_isped)




        return bboxes, image_isped


    def evaluate(self):
        mAP_path = exp_folder + '/mAP'
        if not os.path.exists(mAP_path):
            os.makedirs(mAP_path)

        predicted_dir_path = mAP_path + '/predicted'
        ground_truth_dir_path = mAP_path + '/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        time_total = 0
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)


                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print('=> ground truth of %s:' % image_path)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):

                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_path)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                # bboxes_pr, image_isped = self.predict(image, image_name)
                t1 = time.time()
                bboxes_pr, image_isped = self.predict(image, image_name)
                time_total += time.time() - t1

                if self.write_image:
                    if self.isp_flag:
                        image = utils.draw_bbox(image_isped, bboxes_pr, self.classes, show_label=self.show_label)
                    else:
                        image = utils.draw_bbox(image_isped, bboxes_pr, self.classes, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())

if __name__ == '__main__': YoloTest().evaluate()



