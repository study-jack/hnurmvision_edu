#include "io_context.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <thread>

namespace training::io {
namespace {

// [第二讲 IO] 仿真器发的 roll/pitch/yaw 单位是「度」，按 ZYX 欧拉角转成四元数。
Eigen::Quaterniond quaternion_from_rpy_deg(double roll_deg, double pitch_deg, double yaw_deg) {
  const double k = kPi / 180.0;
  return Eigen::AngleAxisd(yaw_deg * k, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch_deg * k, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll_deg * k, Eigen::Vector3d::UnitX());
}

}  // namespace

// ===== [第二讲 IO/Open] BEGIN：建节点、订阅图像与 IMU、发布云台指令 =====
void IoContext::open() {
  if (!rclcpp::ok()) {
    // 不让 rclcpp 自己接管 SIGINT：否则按 Ctrl+C 时上下文先被 shutdown，
    // close() 里补发的停止指令会被丢掉。信号由 main.cpp 的 handler 处理。
    rclcpp::InitOptions options;
    options.shutdown_on_signal = false;
    rclcpp::init(0, nullptr, options);
    owns_ros_context = true;
  }
  node = std::make_shared<rclcpp::Node>("training_io");

  // 仿真器两端都用 sensor_data QoS（best effort）；这里必须一致，否则收不到任何数据。
  const auto qos = rclcpp::SensorDataQoS();

  image_sub = node->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
        // 仿真器发布 rgb8；main.cpp 的 imshow 按 BGR 理解，所以在这里转一次。
        if (msg->encoding != "rgb8" && msg->encoding != "bgr8") return;
        if (msg->width == 0 || msg->height == 0) return;

        const cv::Mat view(static_cast<int>(msg->height), static_cast<int>(msg->width),
                           CV_8UC3, const_cast<unsigned char*>(msg->data.data()),
                           static_cast<size_t>(msg->step));
        cv::Mat bgr;
        if (msg->encoding == "rgb8") {
          cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);
        } else {
          bgr = view.clone();
        }

        // 图像采集时刻：使用消息 header 的 ROS 时间，扣除「传感器→回调」的传输延迟，
        // 映射到本程序的 steady_clock。直接用 now() 会把传输延迟算进帧间隔，第四讲测速会偏。
        TimePoint stamp = Clock::now();
        const rclcpp::Time header_stamp(msg->header.stamp);
        if (header_stamp.nanoseconds() > 0) {
          const int64_t latency_ns = (node->now() - header_stamp).nanoseconds();
          if (latency_ns > 0 && latency_ns < 1000000000LL) {
            stamp -= std::chrono::nanoseconds(latency_ns);
          }
        }

        std::lock_guard<std::mutex> lock(mutex);
        latest_image = bgr;  // cvtColor/clone 已分配独立内存
        latest_stamp = stamp;
        fresh_image = true;
      });

  recv_sub = node->create_subscription<hnurm_interfaces::msg::VisionRecvData>(
      "/vision_receive_data", qos,
      [this](hnurm_interfaces::msg::VisionRecvData::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex);
        latest_rpy_deg = {msg->roll, msg->pitch, msg->yaw};
        latest_imu_q = quaternion_from_rpy_deg(msg->roll, msg->pitch, msg->yaw);
        latest_bullet_speed = static_cast<double>(msg->bullet_speed.data);
      });

  send_pub = node->create_publisher<hnurm_interfaces::msg::VisionSendData>(
      "/vision_send_data", qos);

  RCLCPP_INFO(node->get_logger(),
              "[第二讲/OPEN] node=training_io subs=[/image_raw,/vision_receive_data] pub=[/vision_send_data]");
}
// ===== [第二讲 IO/Open] END =====

// ===== [第二讲 IO/Read] BEGIN：非阻塞取最新一帧，拼成 SensorData =====
std::optional<SensorData> IoContext::read() {
  if (!node) return std::nullopt;

  rclcpp::spin_some(node);  // 只处理已就绪的回调，不阻塞

  SensorData data;
  std::array<double, 3> rpy{};
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (fresh_image && !latest_image.empty()) {
      data.image = latest_image;  // 浅拷贝：引用计数保证本帧像素在返回后仍然有效
      data.timestamp = latest_stamp;
      data.imu_q = latest_imu_q;
      data.bullet_speed = latest_bullet_speed;
      rpy = latest_rpy_deg;
      data.rpy_deg = latest_rpy_deg;
      latest_image.release();  // 下一帧重新分配，不让两份像素常驻
      fresh_image = false;
    }
  }

  if (data.image.empty()) {
    // 没有新帧：睡 1ms 让出 CPU，主循环不至于 100% 占满一个核
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return std::nullopt;
  }

  // [第二讲/READ] 终端实时打印 pitch / yaw / 弹速，2 Hz（作业里 READ 的演示点）
  const auto now = Clock::now();
  if (last_read_log.time_since_epoch().count() == 0 ||
      now - last_read_log >= std::chrono::milliseconds(500)) {
    last_read_log = now;
    RCLCPP_INFO(node->get_logger(),
                "[第二讲/READ] pitch=%.2f deg yaw=%.2f deg bullet_speed=%.1f m/s image=%dx%d",
                rpy[1], rpy[2], data.bullet_speed, data.image.cols, data.image.rows);
  }
  return data;
}
// ===== [第二讲 IO/Read] END =====

// ===== [第二讲 IO/Write] BEGIN：Command → VisionSendData =====
void IoContext::write(const Command& command) {
  if (!send_pub) return;

  hnurm_interfaces::msg::VisionSendData msg;
  msg.header.stamp = node->now();
  msg.header.frame_id = "gimbal_link";

  // 0 = LOST_TARGET，1 = CONVERGING，2 = FIRE；仿真器只认 0 和 2。
  // config.allow_fire 是最后一道保险：配置关着就永远发不出 FIRE。
  if (!command.control) {
    msg.target_state.data = 0;
  } else if (command.shoot && config.allow_fire) {
    msg.target_state.data = 2;
  } else {
    msg.target_state.data = 1;
  }

  // Command 用弧度，协议用「度」；符号保持 ROBOMASTER 约定，仿真器内部自己翻转 pitch。
  msg.yaw = static_cast<float>(command.yaw * 180.0 / kPi);
  msg.pitch = static_cast<float>(command.pitch * 180.0 / kPi);
  msg.delta_pitch = 0.0f;
  msg.delta_yaw = 0.0f;
  msg.target_distance = 0.0f;  // 不能填 -1，仿真器收到 -1 会直接丢弃整帧
  msg.original_yout = 0.0f;
  msg.vel_x = 0.0f;
  msg.vel_y = 0.0f;
  msg.vel_yaw = 0.0f;
  msg.control_id = 0.0f;
  msg.target_type.data = 0;  // 第二讲不区分目标类型

  // [第二讲/WRITE] 发布限速到 100 Hz：主循环每圈都调 write()，每圈都发会把 DDS 打满
  const auto now = Clock::now();
  if (last_publish.time_since_epoch().count() != 0 &&
      now - last_publish < std::chrono::milliseconds(10)) {
    return;
  }
  last_publish = now;

  send_pub->publish(msg);

  // [第二讲/WRITE] 1 Hz 打印，避免刷屏
  if (last_write_log.time_since_epoch().count() == 0 ||
      now - last_write_log >= std::chrono::seconds(1)) {
    last_write_log = now;
    RCLCPP_INFO(node->get_logger(),
                "[第二讲/WRITE] state=%u yaw=%.2f deg pitch=%.2f deg fire=%s",
                static_cast<unsigned>(msg.target_state.data), msg.yaw, msg.pitch,
                msg.target_state.data == 2 ? "yes" : "no");
  }
}
// ===== [第二讲 IO/Write] END =====

// ===== [第二讲 IO/Close] BEGIN：释放订阅/发布器，必要时关闭 ROS 上下文 =====
void IoContext::close() {
  if (node) {
    RCLCPP_INFO(node->get_logger(), "[第二讲/CLOSE] 发送停止指令，释放订阅/发布器并退出");
    // 退出前必须显式发停止指令：主循环可能在 control=true 的状态下被 Ctrl+C 打断，
    // 只 reset 发布器不会让仿真器停下来。
    if (send_pub) {
      hnurm_interfaces::msg::VisionSendData stop;
      stop.header.stamp = node->now();
      stop.header.frame_id = "gimbal_link";
      stop.target_state.data = 0;  // 0 = 不控制
      stop.target_distance = 0.0f;
      for (int i = 0; i < 3; ++i) {
        send_pub->publish(stop);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    }
  }
  send_pub.reset();
  recv_sub.reset();
  image_sub.reset();
  {
    std::lock_guard<std::mutex> lock(mutex);
    latest_image.release();
    fresh_image = false;
  }
  node.reset();
  if (owns_ros_context && rclcpp::ok()) rclcpp::shutdown();
  owns_ros_context = false;
}
// ===== [第二讲 IO/Close] END =====

}  // namespace training::io
