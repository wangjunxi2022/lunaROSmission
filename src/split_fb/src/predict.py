#!/usr/bin/env python
# coding:utf-8
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

import rospy
from sensor_msgs.msg import Image as type_IMG
from cv_bridge import CvBridge
import cv2



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='unet_carvana_scale0.5_epoch2.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
#     parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
#     return parser.parse_args()



def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def color_image_callback(msg):
    # 将ROS图像消息转换为OpenCV格式
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    img_pil = Image.fromarray(color_image)
    mask = predict_img(net=net,
                           full_img=img_pil,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           device=device)
    # 在OpenCV窗口中显示颜色图像
    result = mask_to_image(mask, mask_values)
    result.save("/home/joonsi/Documents/catkin_ws/src/split_fb/src/b.jpg")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=3, n_classes=2, bilinear=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model unet_carvana_scale0.5_epoch2.pth')
    logging.info(f'Using device {device}')

    net.to(device=device)
    # f = open('unet_carvana_scale0.5_epoch2.pth')
    state_dict = torch.load('/home/joonsi/Documents/catkin_ws/src/split_fb/src/unet_carvana_scale0.5_epoch2.pth', map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    rospy.init_node("display_node")

    # 创建订阅颜色图像和深度图像的ROS话题
    rospy.Subscriber("/camera/color/image_raw", type_IMG, color_image_callback)

    # 循环等待回调
    rospy.spin()
    # for i, filename in enumerate(in_files):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = Image.open(filename)

    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask, mask_values)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
