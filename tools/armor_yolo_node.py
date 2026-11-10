#!/usr/bin/env python3
"""装甲板检测演示节点：/image_raw -> YOLO26n-pose，模型不可用时回落到 lesson03 OpenCV。

运行（需要 ROS 2 Humble 环境）：
  source /opt/ros/humble/setup.bash
  source simulation/install_interfaces/setup.bash
  ROS_DOMAIN_ID=89 <venv>/bin/python tools/armor_yolo_node.py

环境变量：
  ARMOR_WEIGHTS   best.pt 路径
  ARMOR_DETECTOR  yolo | opencv   （默认 yolo，失败自动回落）
"""
from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
except Exception as exc:  # pragma: no cover
    print("无法导入 rclpy/sensor_msgs:", exc, file=sys.stderr)
    raise

WEIGHTS = os.environ.get(
    "ARMOR_WEIGHTS",
    "/home/jack/桌面/yolo26_pose_rmvision-main/提交_任务2/model/best.pt")
FORCE_OPENCV = os.environ.get("ARMOR_DETECTOR", "yolo").lower() == "opencv"
CONF = float(os.environ.get("ARMOR_CONF", "0.25"))
IMGSZ = int(os.environ.get("ARMOR_IMGSZ", "640"))

CLASSES = ["B1", "B2", "B3", "B4", "B5", "BO", "BS",
           "R1", "R2", "R3", "R4", "R5", "RO", "RS"]
LARGE_IDS = {5, 6, 12, 13}


# ===================== YOLO =====================
class YoloDetector:
    def __init__(self, weights: str):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.names = CLASSES
        # 预热
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.model.predict(dummy, imgsz=IMGSZ, conf=CONF, device=0,
                               verbose=False)

    def __call__(self, image: np.ndarray):
        r = self.model.predict(image, imgsz=IMGSZ, conf=CONF, device=0,
                               verbose=False)[0]
        out = []
        boxes = r.boxes
        kpts = r.keypoints
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            corners = None
            if kpts is not None and kpts.xy is not None and i < len(kpts.xy):
                kp = kpts.xy[i].tolist()  # TL, BL, BR, TR
                if len(kp) == 4:
                    # Armor.corners 约定：左上、右上、右下、左下
                    corners = [kp[0], kp[3], kp[2], kp[1]]
            if corners is None:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            out.append({
                "corners": np.array(corners, dtype=np.float32),
                "conf": float(boxes.conf[i]),
                "cls": cls_id,
                "large": cls_id in LARGE_IDS,
            })
        return out


# ===================== OpenCV lesson03 端口 =====================
K_MIN_BAR_ASPECT = 1.3
K_MAX_BAR_ASPECT = 12.0
K_MIN_BAR_AREA = 20.0
K_MAX_BAR_LENGTH = 240.0
K_MAX_BAR_AREA = 1400.0
K_MIN_BAR_FILL = 0.45
K_MAX_LEN_RATIO = 1.6
K_MIN_WIDTH_RATIO = 0.8
K_MAX_WIDTH_RATIO = 2.8
K_LARGE_ARMOR_RATIO = 2.0
K_MIN_PARALLEL = 0.72
K_MAX_HUE_DIFF = 20.0
K_MIN_BAR_BRIGHTNESS = 110.0


def _clamp01(v):
    return max(0.0, min(1.0, v))


def _long_axis(pts):
    best = np.array([1.0, 0.0], np.float32)
    best_len = -1.0
    for i in range(4):
        side = pts[(i + 1) % 4] - pts[i]
        ln = float(np.hypot(side[0], side[1]))
        if ln > best_len:
            best_len, best = ln, side
    if best_len < 1e-3:
        return np.array([0.0, -1.0], np.float32), 0.0
    axis = best / best_len
    if axis[1] > 0:
        axis = -axis
    return axis.astype(np.float32), best_len


def _extract_bars(hsv, color, image_area):
    if color == 0:
        low = cv2.inRange(hsv, (0, 100, 100), (35, 255, 255))
        high = cv2.inRange(hsv, (165, 100, 100), (180, 255, 255))
        mask = cv2.bitwise_or(low, high)
    else:
        mask = cv2.inRange(hsv, (80, 100, 100), (145, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < K_MIN_BAR_AREA or area > K_MAX_BAR_AREA or area > image_area * 0.01:
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(contour)
        if w < 1.0 or h < 1.0:
            continue
        length, width = max(w, h), min(w, h)
        if length < 4.0 or length > K_MAX_BAR_LENGTH:
            continue
        aspect = length / width
        if aspect < K_MIN_BAR_ASPECT or aspect > K_MAX_BAR_ASPECT:
            continue
        fill = area / (w * h + 1e-6)
        if fill < K_MIN_BAR_FILL:
            continue
        if width < 2.0 or width > 25.0:
            continue
        pts = cv2.boxPoints(((cx, cy), (w, h), _)).astype(np.float32)
        axis, axis_len = _long_axis(pts)
        if axis_len < 1e-3:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(hsv.shape[1], x + bw), min(hsv.shape[0], y + bh)
        mean_h = mean_v = 0.0
        if x1 > x0 and y1 > y0:
            m = cv2.mean(hsv[y0:y1, x0:x1], mask[y0:y1, x0:x1])
            mean_h, mean_v = float(m[0]), float(m[2])
        bars.append({
            "center": np.array([cx, cy], np.float32),
            "axis": axis,
            "length": axis_len,
            "width": area / (axis_len + 1e-6),
            "color": color,
            "score": _clamp01(0.4 + 0.2 * (aspect - K_MIN_BAR_ASPECT)
                              + 0.5 * (fill - K_MIN_BAR_FILL)),
            "mean_h": mean_h,
            "mean_v": mean_v,
        })
    return bars


def _armor_corners(left, right):
    ab = right["center"] - left["center"]
    norm = float(np.hypot(ab[0], ab[1]))
    if norm < 1e-3:
        return None
    ab = ab / norm
    na = np.array([-left["axis"][1], left["axis"][0]], np.float32)
    if float(na @ ab) > 0:
        na = -na
    nb = -na
    a_half = left["axis"] * left["length"] * 0.5
    b_half = right["axis"] * right["length"] * 0.5
    a_off = na * left["width"] * 0.5
    b_off = nb * right["width"] * 0.5
    return np.array([
        left["center"] + a_half + a_off,
        right["center"] + b_half + b_off,
        right["center"] - b_half + b_off,
        left["center"] - a_half + a_off,
    ], np.float32)


def _quad_area(c):
    area = 0.0
    for i in range(4):
        p, q = c[i], c[(i + 1) % 4]
        area += p[0] * q[1] - q[0] * p[1]
    return abs(area) * 0.5


def detect_opencv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_area = float(image.shape[0] * image.shape[1])
    bars = _extract_bars(hsv, 0, image_area) + _extract_bars(hsv, 1, image_area)
    if len(bars) < 2:
        return []
    bars.sort(key=lambda b: -b["length"])
    candidates = []
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            a, b = bars[i], bars[j]
            if a["color"] != b["color"]:
                continue
            hue_diff = abs(a["mean_h"] - b["mean_h"])
            hue_diff = min(hue_diff, 180.0 - hue_diff)
            if hue_diff > K_MAX_HUE_DIFF:
                continue
            if min(a["mean_v"], b["mean_v"]) < K_MIN_BAR_BRIGHTNESS:
                continue
            avg_len = (a["length"] + b["length"]) * 0.5
            if avg_len < 6.0:
                continue
            len_ratio = max(a["length"], b["length"]) / min(a["length"], b["length"])
            if len_ratio > K_MAX_LEN_RATIO:
                continue
            delta = b["center"] - a["center"]
            distance = float(np.hypot(delta[0], delta[1]))
            if distance < K_MIN_WIDTH_RATIO * avg_len or distance > K_MAX_WIDTH_RATIO * avg_len:
                continue
            if abs(delta[1]) > 0.8 * distance:
                continue
            parallel = abs(float(a["axis"] @ b["axis"]))
            if parallel < K_MIN_PARALLEL:
                continue
            d = delta / distance
            orthogonal = abs(float(a["axis"][0] * d[1] - a["axis"][1] * d[0]))
            if orthogonal < K_MIN_PARALLEL:
                continue
            a_is_left = a["center"][0] <= b["center"][0]
            corners = _armor_corners(a, b) if a_is_left else _armor_corners(b, a)
            if corners is None or _quad_area(corners) < 80.0:
                continue
            width_ratio = distance / avg_len
            size_term = _clamp01(1.0 - (len_ratio - 1.0) / (K_MAX_LEN_RATIO - 1.0))
            angle_term = _clamp01((parallel - K_MIN_PARALLEL) / (1.0 - K_MIN_PARALLEL))
            if width_ratio < K_LARGE_ARMOR_RATIO:
                width_term = _clamp01((width_ratio - K_MIN_WIDTH_RATIO)
                                      / (K_LARGE_ARMOR_RATIO - K_MIN_WIDTH_RATIO))
            else:
                width_term = _clamp01((K_MAX_WIDTH_RATIO - width_ratio)
                                      / (K_MAX_WIDTH_RATIO - K_LARGE_ARMOR_RATIO))
            conf = _clamp01(0.15 + 0.35 * size_term + 0.30 * angle_term + 0.20 * width_term)
            if conf < 0.45:
                continue
            candidates.append({
                "corners": corners,
                "conf": conf,
                "cls": -1,
                "large": width_ratio > K_LARGE_ARMOR_RATIO,
            })
    candidates.sort(key=lambda c: -c["conf"])
    kept = []
    for cand in candidates:
        box = cv2.boundingRect(cand["corners"].astype(np.int32))
        bx, by, bw, bh = box
        dup = False
        for k in kept:
            kx, ky, kw, kh = cv2.boundingRect(k["corners"].astype(np.int32))
            ix1, iy1 = max(bx, kx), max(by, ky)
            ix2, iy2 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter > 0 and inter > 0.55 * min(bw * bh, kw * kh):
                if bw * bh < kw * kh:
                    kept[kept.index(k)] = cand
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


# ===================== ROS 节点 =====================
class ArmorDemo(Node):
    def __init__(self):
        super().__init__("armor_yolo_demo")
        self.lock = threading.Lock()
        self.frame = None
        self.stamp = 0.0
        self.frames = 0
        self.detector_name = "opencv"
        self.detector = detect_opencv
        if not FORCE_OPENCV:
            try:
                self.detector = YoloDetector(WEIGHTS)
                self.detector_name = "yolo"
                self.get_logger().info(f"YOLO 已加载: {WEIGHTS}")
            except Exception as exc:
                self.get_logger().error(f"YOLO 加载失败，回落 OpenCV: {exc}")
        else:
            self.get_logger().info("ARMOR_DETECTOR=opencv，强制使用 lesson03")
        self.sub = self.create_subscription(
            Image, "/image_raw", self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f"订阅 /image_raw，检测器={self.detector_name}")

    def on_image(self, msg: Image):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = img.copy()
        with self.lock:
            self.frame = img
            self.stamp = time.time()


def draw(canvas, dets, name):
    for d in dets:
        pts = d["corners"].astype(np.int32)
        if name == "yolo":
            color = (255, 140, 0)
            label = f"{CLASSES[d['cls']]} {d['conf']:.2f}"
        else:
            color = (0, 200, 0) if d["large"] else (0, 140, 255)
            label = f"{'L' if d['large'] else 'S'} {d['conf']:.2f}"
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, tuple(p), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (int(pts[0][0]), max(16, int(pts[0][1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    rclpy.init()
    node = ArmorDemo()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    win = "Armor Detection - YOLO (fallback OpenCV)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps, last = 0.0, time.time()
    try:
        while True:
            with node.lock:
                frame, stamp = node.frame, node.stamp
            if frame is None:
                time.sleep(0.01)
                cv2.waitKey(1)
                continue
            t0 = time.time()
            dets = node.detector(frame)
            dt = time.time() - t0
            canvas = frame.copy()
            draw(canvas, dets, node.detector_name)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last))
            last = now
            cv2.putText(canvas, f"{node.detector_name.upper()}  {dt*1000:.1f} ms  "
                               f"{fps:.1f} FPS  armor={len(dets)}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1)
            if key in (27, ord("q")):
                break
            node.frames += 1
            if node.frames % 60 == 0:
                print(f"[{node.detector_name}] frame={node.frames} "
                      f"det={len(dets)} infer={dt*1000:.1f}ms fps={fps:.1f}",
                      flush=True)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
