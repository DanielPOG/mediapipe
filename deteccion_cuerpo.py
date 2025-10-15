import cv2
import time
import numpy as np
from ultralytics import YOLO


def draw_skeleton(frame: np.ndarray, kpts_xy: np.ndarray, conf: np.ndarray | None = None,
                  conf_thresh: float = 0.3) -> None:
    """
    Dibuja el esqueleto COCO (17 puntos) usando coordenadas (x, y).
    kpts_xy: np.ndarray de shape (17, 2)
    conf: np.ndarray de shape (17,) con confidencias opcionales
    """
    # Índices COCO para YOLOv8-Pose (17 KP)
    # 0-nose, 1-l_eye, 2-r_eye, 3-l_ear, 4-r_ear,
    # 5-l_shoulder, 6-r_shoulder, 7-l_elbow, 8-r_elbow, 9-l_wrist, 10-r_wrist,
    # 11-l_hip, 12-r_hip, 13-l_knee, 14-r_knee, 15-l_ankle, 16-r_ankle
    edges = [
        (5, 6),      # hombros
        (5, 7), (7, 9),   # brazo izq
        (6, 8), (8, 10),  # brazo der
        (11, 12),         # cadera
        (5, 11), (11, 13), (13, 15),  # pierna izq
        (6, 12), (12, 14), (14, 16),  # pierna der
        (0, 5), (0, 6)     # cabeza a hombros (opcional)
    ]

    # Dibujar puntos
    for i, (x, y) in enumerate(kpts_xy):
        if x <= 0 or y <= 0:
            continue
        if conf is not None and conf.size == 17 and conf[i] < conf_thresh:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Dibujar conexiones
    for a, b in edges:
        xa, ya = kpts_xy[a]
        xb, yb = kpts_xy[b]
        if min(xa, ya, xb, yb) <= 0:
            continue
        if conf is not None and conf.size == 17 and (conf[a] < conf_thresh or conf[b] < conf_thresh):
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), 2)


def main():
    """
    Detección de cuerpo completo y esqueleto usando YOLOv8-Pose (sin MediaPipe)
    """
    # Carga del modelo de pose (descarga automática en el primer uso)
    model = YOLO('yolov8n-pose.pt')  # rápido; cambiar a 'yolov8s-pose.pt' si quieres más precisión

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: no se pudo abrir la cámara (índice 0).')
        return

    prev = time.time()
    prev_center = None
    movement_text = '—'

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('No se pudo leer de la cámara.')
                break

            # Opcional: reducir tamaño para mayor FPS
            h, w = frame.shape[:2]
            target_w = 960
            if w > target_w:
                scale = target_w / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Inferencia (sin verbose)
            results = model(frame, verbose=False)

            # Tomar primer resultado (batch=1)
            for r in results:
                # Dibujar cajas y esqueletos por cada persona
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None

                if r.keypoints is not None and r.keypoints.xy is not None:
                    k_all = r.keypoints.xy.cpu().numpy()  # (N, 17, 2)
                    k_conf = None
                    try:
                        if r.keypoints.conf is not None:
                            k_conf = r.keypoints.conf.cpu().numpy()  # (N, 17)
                    except Exception:
                        k_conf = None

                    num_people = k_all.shape[0]
                    for i in range(num_people):
                        kpts_xy = k_all[i]
                        conf_i = k_conf[i] if k_conf is not None and i < k_conf.shape[0] else None
                        draw_skeleton(frame, kpts_xy, conf_i, conf_thresh=0.25)

                        # Caja si existe
                        if r.boxes is not None and i < len(r.boxes):
                            x1, y1, x2, y2 = r.boxes.xyxy[i].int().tolist()
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            label = f'Persona {confs[i]:.2f}' if confs is not None else 'Persona'
                            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Centro para movimiento: promedio de puntos válidos
                        visible = kpts_xy[(kpts_xy[:, 0] > 0) & (kpts_xy[:, 1] > 0)]
                        if len(visible) > 0:
                            center = visible.mean(axis=0)
                            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                            if prev_center is not None:
                                dx, dy = center[0] - prev_center[0], center[1] - prev_center[1]
                                dist = np.hypot(dx, dy)
                                if dist > 8:  # umbral de movimiento
                                    if abs(dx) > abs(dy):
                                        movement_text = 'Derecha' if dx > 0 else 'Izquierda'
                                    else:
                                        movement_text = 'Abajo' if dy > 0 else 'Arriba'
                                else:
                                    movement_text = 'Estacionario'
                            prev_center = center

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev) if now > prev else 0.0
            prev = now

            # Overlay de info
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Movimiento: {movement_text}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('YOLOv8 Pose - Cuerpo Completo (sin MediaPipe)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
