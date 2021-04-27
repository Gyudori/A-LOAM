// Author:   Gyuseok Lee            ys96000@naver.com

#include <ios>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

// Added headers
#include <thread>
#include <chrono>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));

    return lidar_data_buffer;
}

int main(int argc, char** argv)
{
    // ROS initiating
    ros::init(argc, argv, "kittiHelperGyu");
    ros::NodeHandle nh;

    // Variable definition
    std::string dataset_folder, sequence_number, output_bag_file;
    int publish_delay, load_frame_num;    
    bool to_bag, load_gt_path;
    pcl::PointCloud<pcl::PointXYZI> lidarMap;

    // Get parameter from launch file
    nh.getParam("dataset_folder", dataset_folder);
    nh.getParam("publish_delay", publish_delay);
    nh.getParam("sequence_number", sequence_number);
    nh.getParam("to_bag", to_bag);
    nh.getParam("load_gt_path", load_gt_path);
    nh.getParam("output_bag_file", output_bag_file);
    nh.getParam("load_frame_num", load_frame_num);

    ros::Publisher pub_laser_cloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);
    ros::Publisher pubOdomGT = nh.advertise<nav_msgs::Odometry> ("/odometry_gt", 5);
    nav_msgs::Odometry odomGT;
    odomGT.header.frame_id = "/camera_init";
    odomGT.child_frame_id = "/ground_truth";
    
    // Timestamp & GT path
    std::string timestamp_path = "data_odometry_calib/dataset/sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);

    std::string ground_truth_path = "data_odometry_poses/dataset/poses/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);    

    // Transformation variables
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Quaterniond q_transform(R_transform);

    ros::Rate r(10.0 / publish_delay);

    rosbag::Bag bag_out;
    if (to_bag)
    {
        bag_out.open(output_bag_file, rosbag::bagmode::Write);
    }

    std::string line;
    int line_num = 0;

    while(ros::ok() && line_num < load_frame_num)
    {        
        if(load_gt_path){
            // Timestamp & GT part
            float timestamp = stof(line);
            
            std::getline(ground_truth_file, line);
            std::stringstream pose_stream(line);
            std::string s;
            Eigen::Matrix<double, 3, 4> gt_pose;

            for(std::size_t i = 0; i < 3; ++i)
            {
                for(std::size_t j = 0; j < 4; ++j)
                {
                    std::getline(pose_stream, s, ' ');
                    gt_pose(i, j) = stof(s);
                }
            }
            std::cout << "GT_pose" << std::endl <<gt_pose << std::endl;
            Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>());
            //Eigen::Quaterniond q = q_transform * q_w_i;
            Eigen::Quaterniond q =  q_w_i;
            q.normalize();
            //Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();
            Eigen::Vector3d t = gt_pose.topRightCorner<3, 1>();  

        odomGT.header.stamp = ros::Time().fromSec(timestamp);
        odomGT.pose.pose.orientation.x = q.x();
        odomGT.pose.pose.orientation.y = q.y();
        odomGT.pose.pose.orientation.z = q.z();
        odomGT.pose.pose.orientation.w = q.w();
        odomGT.pose.pose.position.x = t(0);
        odomGT.pose.pose.position.y = t(1);
        odomGT.pose.pose.position.z = t(2);
        pubOdomGT.publish(odomGT);
        }

        // Lidar part
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << std::setfill('0') << std::setw(10) << line_num << ".txt";       
        std::ifstream lidar_data_file(lidar_data_path.str(), std::ifstream::in);    

        lidar_data_file.seekg(0, ios::end);
        int cloud_size = lidar_data_file.tellg();
        std::cout << "totally " << cloud_size << " points in this lidar frame \n" << std::endl;
        lidar_data_file.seekg(0, std::ios::beg);

        std::vector<Eigen::Vector3d> lidar_points;
        std::vector<float> lidar_intensities;
        pcl::PointCloud<pcl::PointXYZI> laser_cloud;   

        std::string line;
        std::string s;
        while(lidar_data_file >> s){                      
            pcl::PointXYZI point;            
            
            point.x = stof(s);
            lidar_data_file >> s;
            point.y = stof(s);
            lidar_data_file >> s;
            point.z = stof(s);
            lidar_data_file >> s;
            point.intensity = stof(s);

            laser_cloud.push_back(point);            
        }  
        lidar_data_file.close();   

        // std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        // std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n" << std::endl;

        // std::vector<Eigen::Vector3d> lidar_points;
        // std::vector<float> lidar_intensities;
        // pcl::PointCloud<pcl::PointXYZI> laser_cloud;        

        // for(std::size_t i = 0; i < lidar_data.size(); i+=4)
        // {
        //     lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
        //     lidar_intensities.emplace_back(lidar_data[i+3]);

        //     pcl::PointXYZI point;
        //     pcl::PointXYZ point_xyz;
        //     point.x = lidar_data[i];
        //     point.y = lidar_data[i + 1];
        //     point.z = lidar_data[i + 2];
        //     point.intensity = lidar_data[i + 3];
        //     laser_cloud.push_back(point);
        //     lidarMap.push_back(point);
        // }

        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(laser_cloud, laser_cloud_msg);
        //laser_cloud_msg.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "/camera_init";
        pub_laser_cloud.publish(laser_cloud_msg);

        if (to_bag)
        {
            bag_out.write("/velodyne_points", ros::Time::now(), laser_cloud_msg);
        }
        line_num++;

        std::cout << "KittiHelper, published line: " << line_num << std::endl;

        r.sleep();
    }
    bag_out.close();

    std::cout << "KittiHelper Done \n";

    return 0;
}

