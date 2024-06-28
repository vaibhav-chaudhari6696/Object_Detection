import torch
import cv2
import time
from picamera2 import Picamera2

# Load YOLOv5 model (using YOLOv5-Nano)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

# Function to process and visualize detections
def process_frame(frame):
    results = model(frame)
    labels, coords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    person_count = 0
    object_count = 0
    for label, coord in zip(labels, coords):
        x1, y1, x2, y2, conf = coord
        x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
        if conf > 0.5:  # Confidence threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[int(label)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            object_count += 1
            if model.names[int(label)] == 'car':  # Check if detected object is a person
                person_count += 1
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Object Count: {object_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, person_count

# Initialize PiCamera2
picam2 = Picamera2()
picam2.start()

person_count_total = 0
frame_count = 0

while True:
    frame = picam2.capture_array()

    # Convert the frame from RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_count += 1

    # Process frame every 5 seconds
    if frame_count % 150 == 0:  # 30 frames per second * 5 seconds = 150 frames
        frame, person_count = process_frame(frame)
        person_count_total += person_count  # Accumulate total person count

    cv2.imshow('YOLOv5-Nano Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()
picam2.close()

# Display total person count after stopping the webcam feed
print(f"Total person Detected: {person_count_total}")





