#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

void colorImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    try
    {
        // 将ROS图像消息转换为OpenCV格式
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat color_image = cv_ptr->image;

        // 在OpenCV窗口中显示颜色图像
        cv::imshow("Color Image", color_image);
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    try
    {
        // 将ROS图像消息转换为OpenCV格式
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        cv::Mat depth_image = cv_ptr->image;

        // 在OpenCV窗口中显示深度图像
        cv::imshow("Depth Image", depth_image);
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "camera_display_node");

    // 创建ROS节点句柄
    ros::NodeHandle nh;

    // 创建订阅颜色图像和深度图像的ROS话题
    ros::Subscriber color_sub = nh.subscribe("/camera/color/image_raw", 1, colorImageCallback);
    ros::Subscriber depth_sub = nh.subscribe("/camera/depth/image_rect_raw", 1, depthImageCallback);

    // 循环等待回调
    ros::spin();

    return 0;
}

