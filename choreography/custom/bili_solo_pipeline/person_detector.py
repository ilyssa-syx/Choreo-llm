"""
person_detector.py
==================
基于 YOLOv8/YOLO11 的人体检测模块。

检测策略：
  - 使用 ultralytics 的 YOLO 模型（class 0 = person）
  - 对每帧输出人数、各人的 bbox、最大 bbox 面积占比
  - 支持姿态估计（可选，需 yolov8-pose 模型）

若 ultralytics 未安装，退化为"仅返回空检测结果"并警告。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning(
        "ultralytics not installed. Person detection will be unavailable. "
        "Install with: pip install ultralytics"
    )


# ---------------------------------------------------------------------------
# 检测结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """单个检测 bbox"""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    area_ratio: float   # bbox 面积 / 图像总面积

    @property
    def area(self) -> float:
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))


@dataclass
class FrameDetection:
    """单帧检测结果"""
    timestamp: float
    frame_w: int
    frame_h: int
    person_count: int = 0
    bboxes: list[BBox] = field(default_factory=list)
    max_bbox_area_ratio: float = 0.0    # 最大单人 bbox 占画面面积比
    keypoints: Optional[list] = None    # 姿态关键点（可选）


# ---------------------------------------------------------------------------
# 检测器
# ---------------------------------------------------------------------------

class PersonDetector:
    """人体检测器，封装 YOLO 模型。

    Args:
        model_path: YOLO 模型权重路径（如 'yolov8n.pt'）；
                    首次调用时 ultralytics 会自动下载
        conf_threshold: 检测置信度阈值
        enable_pose: 是否启用姿态估计（需使用 yolov8n-pose.pt 等 Pose 模型）
        device: 推理设备（'cpu' / 'cuda' / '0' 等）；None = 自动选择
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        enable_pose: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.enable_pose = enable_pose
        self._model: Optional["YOLO"] = None  # 延迟加载
        self._model_path = model_path
        self._device = device

        if not _YOLO_AVAILABLE:
            logger.error(
                "ultralytics is required for person detection. "
                "Install with: pip install ultralytics"
            )

    def _load_model(self) -> bool:
        """延迟加载模型（首次调用时）。"""
        if self._model is not None:
            return True
        if not _YOLO_AVAILABLE:
            return False
        try:
            logger.info("Loading YOLO model: %s", self._model_path)
            self._model = YOLO(self._model_path)
            if self._device:
                self._model.to(self._device)
            logger.info("YOLO model loaded successfully.")
            return True
        except Exception as exc:
            logger.error("Failed to load YOLO model '%s': %s", self._model_path, exc)
            return False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> FrameDetection:
        """检测单帧中的人体。

        Args:
            frame: BGR numpy array (H, W, 3)
            timestamp: 该帧的时间戳（秒）

        Returns:
            FrameDetection 实例
        """
        h, w = frame.shape[:2]
        det = FrameDetection(timestamp=timestamp, frame_w=w, frame_h=h)

        if not self._load_model():
            return det

        try:
            results = self._model(
                frame,
                conf=self.conf_threshold,
                classes=[0],  # class 0 = person
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO detection failed at t=%.2f: %s", timestamp, exc)
            return det

        if not results:
            return det

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return det

        img_area = w * h
        bboxes: list[BBox] = []
        for box in boxes:
            # xyxy 格式
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            area = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio = area / img_area if img_area > 0 else 0.0
            bboxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, area_ratio=area_ratio))

        bboxes.sort(key=lambda b: b.area_ratio, reverse=True)
        det.bboxes = bboxes
        det.person_count = len(bboxes)
        det.max_bbox_area_ratio = bboxes[0].area_ratio if bboxes else 0.0

        # 可选：姿态关键点
        if self.enable_pose and hasattr(result, "keypoints") and result.keypoints is not None:
            try:
                det.keypoints = result.keypoints.xy.cpu().numpy().tolist()
            except Exception:
                pass

        return det

    def detect_frames(
        self,
        timestamps: list[float],
        frames: list[np.ndarray],
    ) -> list[FrameDetection]:
        """批量检测帧序列。

        Args:
            timestamps: 各帧时间戳列表
            frames: 各帧 BGR numpy array 列表

        Returns:
            FrameDetection 列表，顺序与输入一致
        """
        results: list[FrameDetection] = []
        for ts, frame in zip(timestamps, frames):
            results.append(self.detect_frame(frame, timestamp=ts))
        return results
