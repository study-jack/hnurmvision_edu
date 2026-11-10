#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training {
std::vector<Armor> lesson03_detection(const cv::Mat& image);
std::vector<Armor> lesson03_detection_opencv(const cv::Mat& image);
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state);
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed);
}  // namespace training
