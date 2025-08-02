#pragma once

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

class FlexCloud {
public:
    // --- Method 1: Mirrors select_keyframes.cpp ---
    void run_keyframe_interpolation_from_files(
        const std::string& config_path,
        const std::string& pos_dir_path,
        const std::string& kitti_odom_path,
        const std::string& pcd_dir_path,
        const std::string& dst_dir_path);

    // --- Method 2: Mirrors pcd_georef.cpp ---
    void run_georeferencing_from_files(
        const std::string& config_path,
        const std::string& ref_path,
        const std::string& slam_path,
        const std::string& pcd_path,
        const std::string& pcd_out_path);
};