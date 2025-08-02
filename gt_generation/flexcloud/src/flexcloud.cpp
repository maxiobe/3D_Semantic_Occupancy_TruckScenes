#include "flexcloud/flexcloud.hpp"
// Include other necessary headers from the project
#include "flexcloud/keyframe_interpolation.hpp"
#include "flexcloud/pcd_georef.hpp"

void FlexCloud::run_keyframe_interpolation_from_files(
    const std::string& config_path,
    const std::string& pos_dir_path,
    const std::string& kitti_odom_path,
    const std::string& pcd_dir_path,
    const std::string& dst_dir_path)
{
    // This line is identical to the one in select_keyframes.cpp's main()
    // It creates the object, which runs the entire process in its constructor.
    flexcloud::KeyframeInterpolation set_frames(
        config_path,
        pos_dir_path,
        kitti_odom_path,
        pcd_dir_path,
        dst_dir_path
    );

    // The constructor already saves the results to dst_dir_path,
    // so we don't need to do anything else.
}

void FlexCloud::run_georeferencing_from_files(
    const std::string& config_path,
    const std::string& ref_path,
    const std::string& slam_path,
    const std::string& pcd_path,
    const std::string& pcd_out_path)
{
    // This line is identical to the one in pcd_georef.cpp's main()
    // It creates the object, which runs the entire georeferencing pipeline.
    flexcloud::pcd_georef processor(
        config_path,
        ref_path,
        slam_path,
        pcd_path,
        pcd_out_path
    );
}