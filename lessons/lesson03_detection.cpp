#include "lessons.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace training {
namespace {

// ===== 可调阈值 =====
constexpr float kMinBarAspect = 1.3f;     // 灯条长宽比下限
constexpr float kMaxBarAspect = 18.0f;    // 长宽比上限：场地白/蓝线是细长条，必须排除
constexpr float kMinBarArea = 12.0f;      // 灯条像素面积下限
constexpr float kMaxBarLength = 240.0f;   // 灯条像素长度上限（3 m 处约 30 px）
constexpr float kMaxBarArea = 1400.0f;    // 排除场地线条/大色块，3 m 处灯条约 20-60 px
constexpr float kMinBarFill = 0.35f;      // 轮廓面积 / 最小外接矩形面积
constexpr float kMaxLenRatio = 2.0f;      // 两根灯条长度之比
constexpr float kMaxYDiffRatio = 0.60f;   // 两灯条中心纵向偏差 / 平均灯条长度
constexpr float kMinWidthRatio = 0.3f;    // 灯条间距 / 平均灯条长度
constexpr float kMaxWidthRatio = 8.0f;    // 小目标/透视压缩时两灯条间距会显得很大
constexpr float kLargeArmorRatio = 2.3f;  // 超过此值判定为大装甲板
constexpr float kMinParallel = 0.70f;     // 长轴平行度与正交性下限

struct LightBar {
  cv::Point2f center;
  cv::Point2f axis;   // 长轴单位向量，统一指向上方（图像 y 向下，故 axis.y <= 0）
  float length = 0;   // 长边
  float width = 0;    // 短边
  float score = 0;
  int color = -1;     // 0 红，1 蓝
};

float clamp01(float v) { return std::min(1.0f, std::max(0.0f, v)); }

// 取最小外接矩形的最长边方向作为灯条长轴，避免不同 OpenCV 版本 angle 语义差异。
cv::Point2f longAxis(const cv::Point2f pts[4], float& length) {
  cv::Point2f best(1.0f, 0.0f);
  float best_len = -1.0f;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f side = pts[(i + 1) % 4] - pts[i];
    const float len = std::sqrt(side.x * side.x + side.y * side.y);
    if (len > best_len) {
      best_len = len;
      best = side;
    }
  }
  length = std::max(best_len, 0.0f);
  if (best_len < 1e-3f) return {0.0f, -1.0f};
  cv::Point2f axis(best.x / best_len, best.y / best_len);
  if (axis.y > 0.0f) {  // 统一指向上方
    axis.x = -axis.x;
    axis.y = -axis.y;
  }
  return axis;
}

std::vector<LightBar> extractBars(const cv::Mat& hsv, int color,
                                  float image_area) {
  cv::Mat mask;
  if (color == 0) {
    // 红方装甲：仿真渲染下红/橙灯条落在 H≈0-35 与 165-180 两段。
    cv::Mat low, high;
    cv::inRange(hsv, cv::Scalar(0, 110, 110), cv::Scalar(35, 255, 255), low);
    cv::inRange(hsv, cv::Scalar(165, 110, 110), cv::Scalar(180, 255, 255), high);
    cv::bitwise_or(low, high, mask);
  } else {
    // 蓝方装甲：H≈80-145；灯条实际偏青（H≈90），不能从 92 起。
    cv::inRange(hsv, cv::Scalar(80, 110, 110), cv::Scalar(145, 255, 255), mask);
  }

  const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<LightBar> bars;
  bars.reserve(contours.size());
  for (const auto& contour : contours) {
    const float area = static_cast<float>(cv::contourArea(contour));
    if (area < kMinBarArea || area > kMaxBarArea ||
        area > image_area * 0.01f) continue;

    const cv::RotatedRect rect = cv::minAreaRect(contour);
    const float w = rect.size.width;
    const float h = rect.size.height;
    if (w < 1.0f || h < 1.0f) continue;
    const float length = std::max(w, h);
    const float width = std::min(w, h);
    if (length < 4.0f || length > kMaxBarLength) continue;
    const float aspect = length / width;
    if (aspect < kMinBarAspect || aspect > kMaxBarAspect) continue;
    const float fill = area / (w * h + 1e-6f);
    if (fill < kMinBarFill) continue;

    cv::Point2f pts[4];
    rect.points(pts);
    float axis_len = 0.0f;
    const cv::Point2f axis = longAxis(pts, axis_len);
    if (axis_len < 1e-3f) continue;

    LightBar bar;
    bar.center = rect.center;
    bar.axis = axis;
    bar.length = axis_len;
    bar.width = area / (axis_len + 1e-6f);  // 用面积反推等效宽度，更抗锯齿
    bar.color = color;
    bar.score = clamp01(0.4f + 0.2f * (length / width - kMinBarAspect) +
                        0.5f * (fill - kMinBarFill));
    bars.push_back(bar);
  }
  return bars;
}

std::array<cv::Point2f, 4> armorCorners(const LightBar& left,
                                        const LightBar& right) {
  cv::Point2f ab(right.center.x - left.center.x, right.center.y - left.center.y);
  const float norm = std::sqrt(ab.x * ab.x + ab.y * ab.y);
  if (norm < 1e-3f) return {};
  ab.x /= norm;
  ab.y /= norm;

  // 左侧灯条朝外的法向
  cv::Point2f na(-left.axis.y, left.axis.x);
  if (na.x * ab.x + na.y * ab.y > 0.0f) {
    na.x = -na.x;
    na.y = -na.y;
  }
  const cv::Point2f nb(-na.x, -na.y);

  const cv::Point2f a_half(left.axis.x * left.length * 0.5f,
                           left.axis.y * left.length * 0.5f);
  const cv::Point2f b_half(right.axis.x * right.length * 0.5f,
                           right.axis.y * right.length * 0.5f);
  const cv::Point2f a_off(na.x * left.width * 0.5f, na.y * left.width * 0.5f);
  const cv::Point2f b_off(nb.x * right.width * 0.5f, nb.y * right.width * 0.5f);

  std::array<cv::Point2f, 4> corners;
  corners[0] = left.center + a_half + a_off;   // 左上
  corners[1] = right.center + b_half + b_off;  // 右上
  corners[2] = right.center - b_half + b_off;  // 右下
  corners[3] = left.center - a_half + a_off;   // 左下
  return corners;
}

float quadArea(const std::array<cv::Point2f, 4>& corners) {
  float area = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f& p = corners[i];
    const cv::Point2f& q = corners[(i + 1) % 4];
    area += p.x * q.y - q.x * p.y;
  }
  return std::fabs(area) * 0.5f;
}

cv::Rect quadBox(const std::array<cv::Point2f, 4>& corners) {
  float min_x = corners[0].x, max_x = corners[0].x;
  float min_y = corners[0].y, max_y = corners[0].y;
  for (const auto& p : corners) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }
  return cv::Rect(cv::Point(static_cast<int>(min_x), static_cast<int>(min_y)),
                  cv::Point(static_cast<int>(max_x), static_cast<int>(max_y)));
}

}  // namespace

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  std::vector<Armor> armors;
  if (image.empty() || image.type() != CV_8UC3) return armors;

  cv::Mat hsv;
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
  const float image_area = static_cast<float>(image.total());

  std::vector<LightBar> bars;
  for (int color = 0; color < 2; ++color) {
    const auto candidates = extractBars(hsv, color, image_area);
    bars.insert(bars.end(), candidates.begin(), candidates.end());
  }
  if (bars.size() < 2) return armors;

  std::sort(bars.begin(), bars.end(), [](const LightBar& a, const LightBar& b) {
    return a.length > b.length;
  });

  std::vector<Armor> candidates;
  for (size_t i = 0; i < bars.size(); ++i) {
    for (size_t j = i + 1; j < bars.size(); ++j) {
      const LightBar& a = bars[i];
      const LightBar& b = bars[j];
      if (a.color != b.color) continue;

      const float avg_len = (a.length + b.length) * 0.5f;
      if (avg_len < 6.0f) continue;

      const float len_ratio =
          std::max(a.length, b.length) / std::min(a.length, b.length);
      if (len_ratio > kMaxLenRatio) continue;

      const cv::Point2f delta(b.center.x - a.center.x, b.center.y - a.center.y);
      const float distance = std::sqrt(delta.x * delta.x + delta.y * delta.y);
      if (distance < kMinWidthRatio * avg_len ||
          distance > kMaxWidthRatio * avg_len)
        continue;
      // 装甲板会随车体倾斜，纵向差应相对「两灯条间距」而不是灯条长度。
      if (std::fabs(delta.y) > 0.8f * distance) continue;

      const float parallel = std::fabs(a.axis.x * b.axis.x + a.axis.y * b.axis.y);
      if (parallel < kMinParallel) continue;

      // 灯条长轴应垂直于两条灯条的连线
      const cv::Point2f dir(delta.x / distance, delta.y / distance);
      const float orthogonal = std::fabs(a.axis.x * dir.y - a.axis.y * dir.x);
      if (orthogonal < kMinParallel) continue;

      const bool a_is_left = a.center.x <= b.center.x;
      Armor armor;
      armor.corners = a_is_left ? armorCorners(a, b) : armorCorners(b, a);
      if (quadArea(armor.corners) < 40.0f) continue;

      const float width_ratio = distance / avg_len;
      armor.large_armor = width_ratio > kLargeArmorRatio;
      armor.class_id = -1;  // 传统视觉不做编号识别

      const float size_term = clamp01(1.0f - (len_ratio - 1.0f) /
                                                 (kMaxLenRatio - 1.0f));
      const float angle_term = clamp01((parallel - kMinParallel) /
                                       (1.0f - kMinParallel));
      const float width_term =
          width_ratio < kLargeArmorRatio
              ? clamp01((width_ratio - kMinWidthRatio) /
                        (kLargeArmorRatio - kMinWidthRatio))
              : clamp01((kMaxWidthRatio - width_ratio) /
                        (kMaxWidthRatio - kLargeArmorRatio));
      armor.confidence = clamp01(0.15f + 0.35f * size_term + 0.30f * angle_term +
                                 0.20f * width_term);
      if (armor.confidence < 0.35f) continue;

      candidates.push_back(armor);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Armor& a, const Armor& b) {
              return a.confidence > b.confidence;
            });

  for (const Armor& candidate : candidates) {
    const cv::Rect box = quadBox(candidate.corners);
    bool duplicate = false;
    for (const Armor& kept : armors) {
      const cv::Rect kept_box = quadBox(kept.corners);
      const int inter = (box & kept_box).area();
      if (inter > 0 && inter > 0.55f * std::min(box.area(), kept_box.area())) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) armors.push_back(candidate);
  }
  return armors;
}

}  // namespace training
