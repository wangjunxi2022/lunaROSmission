#!/usr/bin/env python
# coding:utf-8
# 导入所需的ROS和OpenCV库
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def color_image_callback(msg):
    # 将ROS图像消息转换为OpenCV格式
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    # 在OpenCV窗口中显示颜色图像
    cv2.imshow("Color Image", color_image)
    cv2.waitKey(1)

def depth_image_callback(msg):
    # 将ROS图像消息转换为OpenCV格式
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # 在OpenCV窗口中显示深度图像
    cv2.imshow("Depth Image", depth_image)
    cv2.waitKey(1)

def main():
    # 初始化ROS节点
    rospy.init_node("camera_display_node")

    # 创建订阅颜色图像和深度图像的ROS话题
    rospy.Subscriber("/camera/color/image_raw", Image, color_image_callback)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_image_callback)

    # 循环等待回调
    rospy.spin()

if __name__ == '__main__':
    main()

