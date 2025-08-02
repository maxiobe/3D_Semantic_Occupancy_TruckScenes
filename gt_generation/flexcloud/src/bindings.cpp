#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "flexcloud/flexcloud.hpp"

namespace py = pybind11;

PYBIND11_MODULE(flexcloud_bindings, m) {
    m.doc() = "Python bindings for the FlexCloud library";

    py::class_<FlexCloud>(m, "FlexCloud")
        .def(py::init<>())
        // Method 1 (from before)
        .def("run_keyframe_interpolation_from_files",
             &FlexCloud::run_keyframe_interpolation_from_files,
             "Runs the keyframe selection logic by reading from and writing to disk.",
             py::arg("config_path"),
             py::arg("pos_dir_path"),
             py::arg("kitti_odom_path"),
             py::arg("pcd_dir_path"),
             py::arg("dst_dir_path"))

        // --- Method 2 (Updated with py::arg) ---
        .def("run_georeferencing_from_files",
             &FlexCloud::run_georeferencing_from_files,
             "Runs the main georeferencing pipeline from files.",
             py::arg("config_path"),
             py::arg("ref_path"),
             py::arg("slam_path"),
             py::arg("pcd_path"),
             py::arg("pcd_out_path"));
}