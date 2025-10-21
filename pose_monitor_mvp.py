"""MVP: Posture & Pose Monitor con MediaPipe BlazePose
====================================================
Autor: Javi (Mahou San Miguel I+D) – generado con ayuda de ChatGPT
Fecha: 06-jun-2025

Descripción
-----------
• Abre la webcam y procesa cada frame con el pipeline de detección en dos etapas
  (BlazePose Detector → BlazePose GHUM 3D) usando `mediapipe` ≥ 0.10.14.
• Dibuja los 33 keypoints y las conexiones sobre la imagen.
• Calcula un ángulo de flexión cérvico-torácica (ear-shoulder-hip).
• Lanza una alerta visual (“¡Corrige postura!”) si el ángulo supera un umbral
  durante más de `ALERT_DURATION_SEC` segundos (por defecto 3 min).

Dependencias
------------
```bash
python -m venv .venv && source .venv/bin/activate  # opcional
python -m pip install --upgrade mediapipe==0.10.14 opencv-python numpy
```

Uso
---
```bash
python pose_monitor_mvp.py
```
Pulsa **ESC** para salir.

Roadmap breve
-------------
• Sustituir umbral fijo por un modelo LSTM personalizado (aprendizaje del patrón
  postural propio).
• Enviar alertas vía webhook / MQTT en lugar de overlay.
• Añadir `MediaPipe Holistic` para manos y rostro → gestos de pausa o feedback.
• Persistir métricas (CSV, InfluxDB) para analítica longitudinal ESG.

Licencia: Apache-2.0
"""

import time

import cv2
import mediapipe as mp
import numpy as np

# ============================
# Configuración de parámetros
# ============================
POSE_STATIC_IMAGE_MODE = False  # True para imágenes sueltas
MODEL_COMPLEXITY = 2            # 0: lite, 1: full (↓velocidad), 2: heavy (↑precisión)
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

ALERT_ANGLE_THRESHOLD = 40      # grados (flexión cabeza-tronco)
ALERT_DURATION_SEC = 180        # 3 minutos


# ============================
# Clase principal de monitor
# ============================
class PostureMonitor:
    """Procesa frames y gestiona lógica de alerta de mala postura."""

    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=POSE_STATIC_IMAGE_MODE,
            model_complexity=MODEL_COMPLEXITY,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        self.bad_posture_start: float | None = None

    def close(self) -> None:
        """Libera los recursos de MediaPipe Pose."""
        self.pose.close()

    # -------------------------
    # Utilidad para calcular ángulos
    # -------------------------
    @staticmethod
    def _angle(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
        """Devuelve el ángulo en grados en el vértice p2 formado por p1-p2-p3"""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    # -------------------------
    # Procesamiento de cada frame
    # -------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    thickness=2,
                    circle_radius=2,
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    thickness=2,
                ),
            )

            lms = results.pose_landmarks.landmark
            shoulder = (
                lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                lms[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
            )
            ear = (
                lms[self.mp_pose.PoseLandmark.LEFT_EAR].x,
                lms[self.mp_pose.PoseLandmark.LEFT_EAR].y,
            )
            hip = (
                lms[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                lms[self.mp_pose.PoseLandmark.LEFT_HIP].y,
            )

            angle = self._angle(ear, shoulder, hip)
            cv2.putText(
                frame,
                f"Angulo: {angle:.1f}°",
                (30, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if angle > ALERT_ANGLE_THRESHOLD:
                if self.bad_posture_start is None:
                    self.bad_posture_start = time.time()
                elif time.time() - self.bad_posture_start > ALERT_DURATION_SEC:
                    cv2.putText(
                        frame,
                        "¡Corrige postura!",
                        (50, 80),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
            else:
                self.bad_posture_start = None

        return frame


# ============================
# Bucle principal (CLI)
# ============================

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la cámara web; comprueba permisos y dispositivo."
        )

    monitor = PostureMonitor()
    print("Presiona ESC para salir…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # espejo para UX
        frame = monitor.process(frame)

        cv2.imshow("Pose Monitor MVP", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break  # ESC

    monitor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
