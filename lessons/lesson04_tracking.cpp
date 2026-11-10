#include "lessons.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace training {
namespace {

// 装甲板外廓尺寸（m）：小装甲 135x135，大装甲 225x135。
constexpr double kSmallHalfWidth = 0.0675;
constexpr double kLargeHalfWidth = 0.1125;
constexpr double kHalfHeight = 0.0675;

bool solveArmor(const Armor& armor, const CameraCalibration& calib,
                Eigen::Vector3d& p_cam) {
  const double hw = armor.large_armor ? kLargeHalfWidth : kSmallHalfWidth;
  const double hh = kHalfHeight;
  // 与 Armor.corners 顺序一致：左上、右上、右下、左下。
  const float hwf = static_cast<float>(hw);
  const float hhf = static_cast<float>(hh);
  const std::vector<cv::Point3f> obj{{-hwf, -hhf, 0.0f}, {hwf, -hhf, 0.0f},
                                     {hwf, hhf, 0.0f},   {-hwf, hhf, 0.0f}};
  std::vector<cv::Point2f> img;
  img.reserve(4);
  for (const auto& c : armor.corners) img.emplace_back(c.x, c.y);

  cv::Vec3d rvec, tvec;
  bool ok = false;
  try {
    ok = cv::solvePnP(obj, img, calib.matrix, calib.distortion, rvec, tvec,
                      false, cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception&) {
    ok = false;
  }
  if (!ok) ok = cv::solvePnP(obj, img, calib.matrix, calib.distortion, rvec,
                             tvec, false, cv::SOLVEPNP_ITERATIVE);
  if (!ok || !std::isfinite(tvec[0]) || !std::isfinite(tvec[1]) ||
      !std::isfinite(tvec[2]))
    return false;
  // 目标必须在相机前方且距离合理，过滤 PnP 翻面解。
  if (tvec[2] < 0.3 || tvec[2] > 25.0) return false;
  p_cam = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
  return true;
}

}  // namespace

std::optional<Target> lesson04_tracking(
    const std::vector<Armor>& armors, TimePoint timestamp,
    const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration,
    TrackingState& state) {
  if (!calibration.valid() || armors.empty()) {
    if (state.valid) ++state.lost_frames;
    if (state.lost_frames > 10) state.valid = false;
    return std::nullopt;
  }

  // 候选：每块装甲做 PnP，转到 IMU 稳定系。
  struct Obs {
    Eigen::Vector3d p;
    double conf;
    double px_width;
  };
  std::vector<Obs> observations;
  observations.reserve(armors.size());
  for (const auto& armor : armors) {
    Eigen::Vector3d p_cam;
    if (!solveArmor(armor, calibration, p_cam)) continue;
    const Eigen::Vector3d p_gimbal =
        calibration.camera_to_gimbal * p_cam + calibration.camera_to_gimbal_translation;
    const double px_width = std::hypot(armor.corners[1].x - armor.corners[0].x,
                                       armor.corners[1].y - armor.corners[0].y);
    observations.push_back({imu_q * p_gimbal, armor.confidence, px_width});
  }
  if (observations.empty()) {
    if (state.valid) ++state.lost_frames;
    return std::nullopt;
  }

  // 选目标：已有跟踪时取离预测位置最近的，否则取置信度最高的。
  Eigen::Vector3d predicted = Eigen::Vector3d::Zero();
  if (state.valid) predicted = state.position + state.velocity * 0.05;
  const Obs* best = nullptr;
  double best_score = -1e9;
  for (const auto& obs : observations) {
    const double score =
        state.valid ? -((obs.p - predicted).norm()) + 0.2 * obs.conf : obs.conf;
    if (score > best_score) {
      best_score = score;
      best = &obs;
    }
  }
  if (best == nullptr) return std::nullopt;

  const Eigen::Vector3d measured = best->p;

  // 一阶低通（alpha-beta）：位置平滑、速度差分。
  if (!state.valid) {
    state.position = measured;
    state.velocity = Eigen::Vector3d::Zero();
    state.valid = true;
  } else {
    const double dt =
        std::chrono::duration<double>(timestamp - state.last_stamp).count();
    const Eigen::Vector3d pred = state.position + state.velocity * dt;
    const Eigen::Vector3d residual = measured - pred;
    if (dt <= 0.0 || dt > 0.5 || residual.norm() > 1.5) {
      // 时间跳变或目标跳变：重新起步，不把跳变当速度。
      state.position = measured;
      state.velocity = Eigen::Vector3d::Zero();
    } else {
      constexpr double kAlpha = 0.7;
      constexpr double kBeta = 0.2;
      state.position = pred + kAlpha * residual;
      state.velocity += (kBeta / dt) * residual;
      if (state.velocity.norm() > 8.0) state.velocity *= 8.0 / state.velocity.norm();
    }
  }
  state.last_stamp = timestamp;
  state.lost_frames = 0;

  static auto last_log = Clock::now();
  if (Clock::now() - last_log > std::chrono::seconds(1)) {
    last_log = Clock::now();
    std::fprintf(stderr, "[track] p=(%.2f, %.2f, %.2f) |p|=%.2f v=%.2f w=%.0fpx class=%d\n",
                 state.position.x(), state.position.y(), state.position.z(),
                 state.position.norm(), state.velocity.norm(), best->px_width, 0);
  }

  Target target;
  target.position = state.position;
  target.velocity = state.velocity;
  return target;
}

}  // namespace training
