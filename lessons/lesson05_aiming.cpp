#include "lessons.hpp"

#include <cmath>

namespace training {
namespace {

constexpr double kGravity = 9.8;

// 空中飞行时目标按匀速外推，迭代收敛落点。
Eigen::Vector3d predictImpact(const Target& target, double bullet_speed,
                              double& flight_time) {
  Eigen::Vector3d aim = target.position;
  double t = aim.norm() / bullet_speed;
  for (int i = 0; i < 5; ++i) {
    t = aim.norm() / bullet_speed;
    aim = target.position + target.velocity * t;
  }
  flight_time = t;
  return aim;
}

}  // namespace

Command lesson05_aiming(const Target& target, TimePoint timestamp,
                        double bullet_speed) {
  (void)timestamp;
  Command command;
  if (!target.position.allFinite() || !target.velocity.allFinite() ||
      !std::isfinite(bullet_speed) || bullet_speed < 5.0)
    return command;

  double flight_time = 0.0;
  const Eigen::Vector3d aim = predictImpact(target, bullet_speed, flight_time);
  const double horizontal = std::hypot(aim.x(), aim.y());
  if (horizontal < 0.3 || aim.z() < -1.0 || aim.z() > 5.0) return command;

  // 弹道方程：tan(theta) = (v^2 - sqrt(v^4 - g(g d^2 + 2 z v^2))) / (g d)
  const double v2 = bullet_speed * bullet_speed;
  const double disc =
      v2 * v2 - kGravity * (kGravity * horizontal * horizontal + 2.0 * aim.z() * v2);
  if (disc < 0.0) return command;  // 超出射程

  const double pitch = std::atan2(v2 - std::sqrt(disc), kGravity * horizontal);
  const double yaw = std::atan2(aim.y(), aim.x());
  if (!std::isfinite(yaw) || !std::isfinite(pitch)) return command;

  command.control = true;
  command.yaw = yaw;
  command.pitch = pitch;
  command.shoot = true;  // 最终仍由 IO 层的 allow_fire 门控
  return command;
}

}  // namespace training
