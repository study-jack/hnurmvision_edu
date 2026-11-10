#include "lessons.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace training {
namespace {

constexpr int kInput = 640;
constexpr float kConfThreshold = 0.25f;
constexpr float kNmsIou = 0.45f;
constexpr int kClasses = 14;
constexpr int kKeypoints = 4;
constexpr int kRowStride = 8400;  // 640/8 + 640/16 + 640/32 三层输出拼接

bool isLargeClass(int id) {
  return id == 5 || id == 6 || id == 12 || id == 13;  // BO/BS/RO/RS
}

struct Letterbox {
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

Letterbox letterbox(const cv::Mat& image, cv::Mat& out) {
  Letterbox lb;
  lb.scale = std::min(kInput / static_cast<float>(image.cols),
                      kInput / static_cast<float>(image.rows));
  const int new_w = std::max(1, static_cast<int>(std::round(image.cols * lb.scale)));
  const int new_h = std::max(1, static_cast<int>(std::round(image.rows * lb.scale)));
  lb.pad_x = (kInput - new_w) * 0.5f;
  lb.pad_y = (kInput - new_h) * 0.5f;

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(new_w, new_h));
  out = cv::Mat(kInput, kInput, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(static_cast<int>(lb.pad_x),
                              static_cast<int>(lb.pad_y), new_w, new_h)));
  return lb;
}

struct Candidate {
  cv::Rect2f box;
  float score = 0;
  int cls = -1;
  cv::Point2f kps[kKeypoints];
};

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);
  const float iw = std::max(0.0f, x2 - x1);
  const float ih = std::max(0.0f, y2 - y1);
  const float inter = iw * ih;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0 ? inter / uni : 0.0f;
}

class YoloEngine {
 public:
  explicit YoloEngine(const std::string& model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "armor_yolo"),
        memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
    ready_ = true;
  }

  bool ready() const { return ready_; }

  std::vector<Armor> run(const cv::Mat& image) {
    std::vector<Armor> armors;
    if (!ready_ || image.empty() || image.type() != CV_8UC3) return armors;

    cv::Mat canvas;
    const Letterbox lb = letterbox(image, canvas);

    std::vector<float> input(static_cast<size_t>(3) * kInput * kInput);
    const int plane = kInput * kInput;
    for (int y = 0; y < kInput; ++y) {
      const cv::Vec3b* row = canvas.ptr<cv::Vec3b>(y);
      for (int x = 0; x < kInput; ++x) {
        const cv::Vec3b& p = row[x];  // BGR -> RGB, /255
        input[0 * plane + y * kInput + x] = p[2] / 255.0f;
        input[1 * plane + y * kInput + x] = p[1] / 255.0f;
        input[2 * plane + y * kInput + x] = p[0] / 255.0f;
      }
    }

    const std::array<int64_t, 4> shape{1, 3, kInput, kInput};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_, input.data(), input.size(), shape.data(), shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                 output_names, 1);
    const float* data = outputs[0].GetTensorData<float>();

    // 输出 [1, 4+14+8, 8400]：cx,cy,w,h + 类别分数 + 4 个角点(x,y)
    std::vector<Candidate> candidates;
    candidates.reserve(64);
    const int kp_offset = 4 + kClasses;
    for (int i = 0; i < kRowStride; ++i) {
      int best_cls = -1;
      float best_score = 0.0f;
      for (int c = 0; c < kClasses; ++c) {
        const float s = data[(4 + c) * kRowStride + i];
        if (s > best_score) {
          best_score = s;
          best_cls = c;
        }
      }
      if (best_score < kConfThreshold) continue;

      const float cx = data[0 * kRowStride + i];
      const float cy = data[1 * kRowStride + i];
      const float w = data[2 * kRowStride + i];
      const float h = data[3 * kRowStride + i];
      Candidate cand;
      cand.score = best_score;
      cand.cls = best_cls;
      cand.box = cv::Rect2f((cx - w * 0.5f - lb.pad_x) / lb.scale,
                            (cy - h * 0.5f - lb.pad_y) / lb.scale,
                            w / lb.scale, h / lb.scale);
      for (int k = 0; k < kKeypoints; ++k) {
        const float kx = data[(kp_offset + 2 * k) * kRowStride + i];
        const float ky = data[(kp_offset + 2 * k + 1) * kRowStride + i];
        cand.kps[k] = cv::Point2f((kx - lb.pad_x) / lb.scale,
                                  (ky - lb.pad_y) / lb.scale);
      }
      candidates.push_back(cand);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                return a.score > b.score;
              });

    std::vector<Candidate> kept;
    for (const auto& cand : candidates) {
      bool suppressed = false;
      for (const auto& k : kept) {
        if (iou(cand.box, k.box) > kNmsIou) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) kept.push_back(cand);
      if (kept.size() >= 32) break;
    }

    const float max_x = static_cast<float>(image.cols - 1);
    const float max_y = static_cast<float>(image.rows - 1);
    for (const auto& c : kept) {
      Armor armor;
      // 训练标签关键点顺序：左上、左下、右下、右上
      // Armor 约定：左上、右上、右下、左下
      const int order[4] = {0, 3, 2, 1};
      for (int k = 0; k < 4; ++k) {
        armor.corners[k].x = std::clamp(c.kps[order[k]].x, 0.0f, max_x);
        armor.corners[k].y = std::clamp(c.kps[order[k]].y, 0.0f, max_y);
      }
      armor.class_id = c.cls;
      armor.confidence = c.score;
      armor.large_armor = isLargeClass(c.cls);
      armors.push_back(armor);
    }
    return armors;
  }

 private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_;
  std::string input_name_;
  std::string output_name_;
  bool ready_ = false;
};

const char* modelPath() {
  const char* env = std::getenv("ARMOR_ONNX");
  return (env && *env) ? env : "assets/armor_pose.onnx";
}

bool forceOpenCV() {
  const char* env = std::getenv("ARMOR_DETECTOR");
  return env && std::string(env) == "opencv";
}

YoloEngine& engine() {
  static YoloEngine instance(modelPath());
  return instance;
}

}  // namespace

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  static bool warned = false;
  if (forceOpenCV()) return lesson03_detection_opencv(image);

  try {
    YoloEngine& e = engine();
    if (!e.ready()) return lesson03_detection_opencv(image);
    return e.run(image);
  } catch (const std::exception& exc) {
    if (!warned) {
      std::fprintf(stderr, "[armor] YOLO 初始化/推理失败，回落 OpenCV: %s\n",
                   exc.what());
      warned = true;
    }
    return lesson03_detection_opencv(image);
  }
}

}  // namespace training
