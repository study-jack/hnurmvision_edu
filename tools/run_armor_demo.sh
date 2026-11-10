#!/usr/bin/env bash
# 一键启动装甲板检测演示：仿真器（若未运行）+ YOLO 实时检测窗口
# 用法: ./tools/run_armor_demo.sh
# 可选: ARMOR_DETECTOR=opencv 强制走 lesson03 兜底
set -eo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${ARMOR_VENV:-/home/jack/桌面/yolo26_pose_rmvision-main/.venv-cuda}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-89}
export DISPLAY=${DISPLAY:-:1}

source /opt/ros/humble/setup.bash
source "$REPO_ROOT/simulation/install_interfaces/setup.bash"

if ! pgrep -f 'bin/daedalus' > /dev/null; then
  echo "仿真器未运行，正在启动..."
  (cd "$REPO_ROOT" && setsid nohup ./scripts/run_simulator.sh > /tmp/armor_sim.log 2>&1 < /dev/null &)
  for _ in $(seq 1 20); do
    pgrep -f 'bin/daedalus' > /dev/null && break
    sleep 0.5
  done
  sleep 3
else
  echo "仿真器已在运行"
fi

exec "$VENV/bin/python" "$REPO_ROOT/tools/armor_yolo_node.py" "$@"
