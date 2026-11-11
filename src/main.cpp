#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <csignal>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
}

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
    if (input) {
      const auto armors = training::lesson03_detection(input->image);
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
    }
    if (interrupted) break;
    devices.write(command);

    if (input && devices.config.display) {
      cv::imshow("RoboMaster training", input->image);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }
  }

  devices.close();
}
