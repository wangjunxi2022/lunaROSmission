#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PointCloud Viewer"));

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
    viewer->spinOnce();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_display_node");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/rslidar_points", 1, cloudCallback);

    while (ros::ok() && !viewer->wasStopped())
    {
        ros::spinOnce();
    }

    return 0;
}