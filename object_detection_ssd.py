from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")
obj = ["cell phone", "calculator", "dog", "clock", "cup"]
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  # Default assumption
print(f"Camera FPS: {fps}")

time.sleep(0.5)

frame_skip = 2  # For now, this seems best set to 2
frame_num = 0
last_detections = []

while True:
    success, frame = cap.read()
    if not success:
        print("video end")
        break

    frame_num += 1

    if frame_num % frame_skip == 0:
        detections = model(frame)
        last_detections = []

        for result in detections:
            for box in result.boxes:
                cls_idx = int(box.cls[0])
                confidence = float(box.conf[0])
                obj_name = model.names.get(cls_idx, "unknown")

                print(f"detected: {obj_name} conf: {confidence}")

                if obj_name in obj and confidence > 0.3:
                    coords = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, coords)
                    last_detections.append((obj_name, confidence, x1, y1, x2, y2))
                    print(f"tracking {obj_name} - confidence: {confidence:.3f}")

    for det in last_detections:
        obj_name, confidence, x1, y1, x2, y2 = det
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{obj_name} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Proctored Test", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("quit")
        break

cap.release()
cv2.destroyAllWindows()
