#pragma once
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <array>
#include <chrono>

namespace training {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
constexpr double kPi = 3.14159265358979323846;

// IO 提供的算法输入；错误、连接状态和设备模式留在 IO 层。
struct CameraCalibration {
  cv::Mat matrix; // 3x3 CV_64F，相机光学坐标系
  cv::Mat distortion;
  Eigen::Matrix3d camera_to_gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_to_gimbal_translation = Eigen::Vector3d::Zero();
  unsigned width=0, height=0;
  bool valid() const { return !matrix.empty(); }
};

struct SensorData {
  cv::Mat image;  // CV_8UC3，BGR，独立拥有像素内存
  TimePoint timestamp{};
  Eigen::Quaterniond imu_q = Eigen::Quaterniond::Identity();
  std::array<double, 3> rpy_deg{};  // 仿真原始 roll/pitch/yaw（度），用于闭环校正
  double bullet_speed = 0;  // m/s
};

// 第三讲输出：一块装甲板的图像观测。
struct Armor {
  std::array<cv::Point2f, 4> corners{};  // 左上、右上、右下、左下
  bool large_armor = false;
  int class_id = -1;
  double confidence = 0;
};

// 第四讲输出：当前观测时刻的单目标状态。无目标用 std::nullopt 表示。
struct Target {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();  // m，IMU稳定系中相对云台原点的目标向量
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  // m/s，同一坐标系
};

// 第五讲输出：绝对角使用 rad，IO 边界统一转换单位和方向。
struct Command {
  bool control = false;
  bool shoot = false;
  double yaw = 0;
  double pitch = 0;
};
}  // namespace training
