#pragma once

#include "types.hpp"

namespace training {
// 仅第四讲使用：上一帧的解算结果，用于差分测速和短时丢帧保持。
struct TrackingState {
  bool valid = false;               // 是否已有可用的上一帧目标
  TimePoint last_stamp{};           // 上一帧图像时刻
  Eigen::Vector3d position = Eigen::Vector3d::Zero();  // IMU 稳定系，m
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  // IMU 稳定系，m/s
  int lost_frames = 0;              // 连续未观测帧数
};
}  // namespace training
