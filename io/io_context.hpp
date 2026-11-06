#pragma once
#include "config.hpp"
#include "types.hpp"
#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <array>
#include <mutex>
#include <optional>

namespace training::io {
struct IoContext {
  Config config;

  // ===== [第二讲 IO/Open] BEGIN：节点、订阅与发布器 =====
  rclcpp::Node::SharedPtr node;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
  rclcpp::Subscription<hnurm_interfaces::msg::VisionRecvData>::SharedPtr recv_sub;
  rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr send_pub;
  bool owns_ros_context = false;  // 只有 open() 自己 rclcpp::init() 时才在 close() 里 shutdown
  // ===== [第二讲 IO/Open] END =====

  // ===== [第二讲 IO/Read] BEGIN：回调写、主循环读的缓存 =====
  std::mutex mutex;
  cv::Mat latest_image;                    // 已转成 BGR，独立拥有像素内存
  bool fresh_image = false;
  TimePoint latest_stamp{};
  Eigen::Quaterniond latest_imu_q = Eigen::Quaterniond::Identity();
  double latest_bullet_speed = 0.0;
  std::array<double, 3> latest_rpy_deg{};  // roll, pitch, yaw（仿真器原始值，单位度）
  // ===== [第二讲 IO/Read] END =====

  // ===== [第二讲 IO/Write] BEGIN：日志限流状态 =====
  TimePoint last_read_log{};
  TimePoint last_write_log{};
  TimePoint last_publish{};  // 发布限速：主循环 ~1kHz，没必要每圈都发
  // ===== [第二讲 IO/Write] END =====

  void open();
  std::optional<SensorData> read();
  void write(const Command& command);
  void close();
};
}  // namespace training::io
