#include "normal.h"

void normal::getpixel(float* a, float* b,float zc,float fx,float fy,float cx,float cy){
	*a = int(*a * fx / zc + cx);
	*b = int(*b * fy / zc + cy);
	return;
}

void normal::getXYZ(float* a, float* b,float zc,float fx,float fy,float cx,float cy){

	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	*a = (*a - cx) * zc * inv_fx;
	*b = (*b - cy) * zc * inv_fy;
	return;
}
normal::normal(){
	NodeHandle nh;
	pc_input.reset(new PointCloud<PointXYZRGB>());
	pc_filter.reset(new PointCloud<PointXYZRGB>());
	result.reset(new PointCloud<PointXYZRGB>());

	tree.reset(new pcl::search::KdTree<pcl::PointXYZRGB>());

	// sensor_msgs::CameraInfo::ConstPtr msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/color/camera_info",ros::Duration());
	// fx = msg->P[0];
	// fy = msg->P[5];
	// cx = msg->P[2];
	// cy = msg->P[6];
	fx = 616.918;
	fy = 616.369;
	cx = 315.43;
	cy = 222.87;

	coefficients.reset(new pcl::ModelCoefficients);
	inliers.reset(new pcl::PointIndices);

	seg1.setOptimizeCoefficients (true);
	seg1.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);   // pcl::SACMODEL_PLANE   SACMODEL_PERPENDICULAR_PLANE
	seg1.setMethodType (pcl::SAC_RANSAC);
	seg1.setMaxIterations (100);								// 200 
	seg1.setDistanceThreshold (0.35);

	axis = Eigen::Vector3f(0.0,0.0,1.0);	 // y : 1.0
	seg1.setAxis(axis);
	seg1.setEpsAngle(  50.0f * (PI/180.0f) );


	seg2.setOptimizeCoefficients (true);
	seg2.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);   // pcl::SACMODEL_PLANE   SACMODEL_PERPENDICULAR_PLANE
	seg2.setMethodType (pcl::SAC_RANSAC);
	seg2.setDistanceThreshold (0.025);
	seg2.setAxis(axis);
	seg2.setEpsAngle(  50.0f * (PI/180.0f) );

	cout << fx << endl;
	cout << fy << endl;
	cout << cx << endl;
	cout << cy << endl;
	service = nh.advertiseService("normal_cb", &normal::process_node, this);

}
bool normal::process_node(pcl_exercise::norm::Request &req, 
						pcl_exercise::norm::Response &res){
	Mat img = imread(req.path, CV_LOAD_IMAGE_UNCHANGED);
	Mat color_img(480, 640, CV_8UC3, Scalar(0,0,0));

	if (img.rows == 0){
		cout << req.path << " can't read image\n";
		return false;
	}
	for( int nrow = 0; nrow < img.rows; nrow++){   			// 480
		for(int ncol = 0; ncol < img.cols; ncol++){   		// 640
			if (img.at<unsigned short int>(nrow,ncol) > 1){
				pcl::PointXYZRGB point;
				float* x = new float(nrow);
				float* y = new float(ncol);
			 	float z = float(img.at<unsigned short int>(nrow,ncol))/1000.;
				getXYZ(y,x,z,fx,fy,cx,cy);
				point.x = z;
				point.y = -*y;
				point.z = -*x;
				pc_input->points.push_back(point);
				free(x);
				free(y);
			} 
		} 
	}
	cout << "get pc\n";
	res.depth = *(cv_bridge::CvImage(std_msgs::Header(), "16UC1", img).toImageMsg());
	pc_input->width = pc_input->points.size();
	pc_input->height = 1;

	seg1.setInputCloud (pc_input);
	seg1.segment (*inliers, *coefficients);

	extract.setInputCloud (pc_input);
	extract.setIndices (inliers);
	extract.setNegative (false);
	extract.filter (*pc_input);

	seg2.setInputCloud (pc_input);
	seg2.segment (*inliers, *coefficients);

	// remove outlier 
	extract.setInputCloud (pc_input);
	extract.setIndices (inliers);
	extract.setNegative (true);
	extract.filter (*pc_input);

	ne.setInputCloud (pc_input);
	ne.setSearchMethod (tree);

	// Output datasets
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch (0.03);
	ne.compute (*cloud_normals);

	cout << cloud_normals->points.size() << endl;


	// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	// viewer->setBackgroundColor (0, 0, 0);
	// pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pc_input);
	// viewer->addPointCloud<pcl::PointXYZRGB> (pc_input, rgb, "sample cloud");
	// viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	// viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (pc_input, cloud_normals, 10, 0.05, "normals");
	// //viewer->addCoordinateSystem (0.1);
	// viewer->initCameraParameters ();

	// while (!viewer->wasStopped ())
	// {
	//   viewer->spinOnce (100);
	//   boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	// }
	
	for (int i = 0; i < pc_input->points.size(); i++){
		float* x = new float(- pc_input->points[i].z);
		float* y = new float(- pc_input->points[i].y);
		getpixel(y,x,pc_input->points[i].x,fx,fy,cx,cy);
		color_img.at<Vec3b>(*x,*y)[0] = abs(cloud_normals->points[i].normal_x * 255);
		color_img.at<Vec3b>(*x,*y)[1] = abs(cloud_normals->points[i].normal_y * 255);
		color_img.at<Vec3b>(*x,*y)[2] = abs(cloud_normals->points[i].normal_z * 255);
		free(x);
		free(y);
	}
	cloud_normals->clear();
	pc_input->clear();
	res.normal = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_img).toImageMsg());
	return true;
}

int main(int argc, char** argv){
	init(argc, argv, "normal");
	normal normal;
	spin();
	return 0;
}