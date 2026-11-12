#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <vector>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// YOLO 输出的类别名；OpenCV 兜底时 class_id=-1。
const char* armorClassName(int id) {
  static const char* names[14] = {"B1", "B2", "B3", "B4", "B5", "BO", "BS",
                                  "R1", "R2", "R3", "R4", "R5", "RO", "RS"};
  return (id >= 0 && id < 14) ? names[id] : nullptr;
}

// 把第三讲检测到的装甲板画到画面上：绿色=大装甲板，橙色=小装甲板。
void draw_armors(cv::Mat& canvas, const std::vector<training::Armor>& armors) {
  for (const auto& armor : armors) {
    std::vector<cv::Point> points;
    points.reserve(armor.corners.size());
    for (const auto& corner : armor.corners) {
      points.emplace_back(cv::Point(cvRound(corner.x), cvRound(corner.y)));
    }
    const cv::Scalar color =
        armor.large_armor ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    cv::polylines(canvas, points, true, color, 2, cv::LINE_AA);
    for (const auto& point : points) {
      cv::circle(canvas, point, 3, color, -1, cv::LINE_AA);
    }
    char label[64];
    const char* name = armorClassName(armor.class_id);
    if (name != nullptr) {
      std::snprintf(label, sizeof(label), "%s %.2f", name, armor.confidence);
    } else {
      std::snprintf(label, sizeof(label), "%s %.2f",
                    armor.large_armor ? "L" : "S", armor.confidence);
    }
    const cv::Point anchor = points.empty() ? cv::Point(0, 0) : points[0];
    cv::putText(canvas, label, cv::Point(anchor.x, std::max(12, anchor.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
  }
}
}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  while (!interrupted) {
    const auto input = devices.read();
    training::Command command{};
    std::vector<training::Armor> armors;
    if (input) {
      const auto det_begin = training::Clock::now();
      armors = training::lesson03_detection(input->image);
      const double det_ms = std::chrono::duration<double, std::milli>(
                                training::Clock::now() - det_begin).count();
      static double det_acc = 0.0;
      static int det_count = 0;
      static int det_total = 0;
      static auto det_report = det_begin;
      det_acc += det_ms;
      ++det_count;
      det_total += static_cast<int>(armors.size());
      if (training::Clock::now() - det_report > std::chrono::seconds(2)) {
        std::fprintf(stderr, "[armor] %.1f ms/frame, %.2f det/frame, %.1f FPS\n",
                     det_acc / std::max(1, det_count),
                     static_cast<double>(det_total) / std::max(1, det_count),
                     1000.0 * det_count / std::max(1.0, det_acc));
        det_acc = 0.0;
        det_count = 0;
        det_total = 0;
        det_report = training::Clock::now();
      }
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
    }
    if (interrupted) break;
    devices.write(command);

    if (input && devices.config.display) {
      cv::Mat canvas = input->image.clone();
      draw_armors(canvas, armors);
      cv::imshow("RoboMaster training", canvas);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }
  }

  devices.close();
}
