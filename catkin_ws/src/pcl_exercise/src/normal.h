#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
// Ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
// Pcl load and ros
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
// Pcl icp
#include <pcl/registration/icp.h>
// Pcl passthrough
#include <pcl/filters/passthrough.h>
// Pcl outlier removal
#include <pcl/filters/statistical_outlier_removal.h>
// Pcl downsampling
#include <pcl/filters/voxel_grid.h>
// Pcl plane filter
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
// Pcl normal
#include <pcl/features/normal_3d.h>

#include "pcl_exercise/norm.h"

#include <pcl/visualization/pcl_visualizer.h>

using namespace ros;
using namespace pcl;
using namespace std;
using namespace cv;
const double PI  =3.141592653589793238463;

class normal{
  public:
	normal();
	bool process_node(pcl_exercise::norm::Request &req, pcl_exercise::norm::Response &res);
	void getXYZ(float* ,float* ,float ,float ,float ,float ,float );
	void getpixel(float* ,float* ,float ,float ,float ,float ,float );
  private:
  	float fx;
  	float fy;
  	float cx;
  	float cy;
  	ros::ServiceServer service;

  	string path;

	PointCloud<PointXYZRGB>::Ptr pc_input;
	PointCloud<PointXYZRGB>::Ptr pc_filter;
	PointCloud<PointXYZRGB>::Ptr result;

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    pcl::SACSegmentation<PointXYZRGB> seg1;
    pcl::SACSegmentation<PointXYZRGB> seg2;
    Eigen::Vector3f axis;
    pcl::PointIndices::Ptr inliers;
    pcl::ModelCoefficients::Ptr coefficients;

	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree;

	sensor_msgs::PointCloud2 ros_cloud_msg;
	sensor_msgs::PointCloud2 origin_map;


	Publisher marker_pub;
	uint32_t shape;
	visualization_msgs::Marker marker;
};