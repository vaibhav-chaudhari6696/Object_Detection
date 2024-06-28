import cv2
from picamera2 import Picamera2

from ultralytics import YOLO,solutions


# def process_frame(frame,classes_to_count):
#     # # Run YOLOv8 inference on the frame
#     # results = model(frame)
#     # # Visualize the results on the frame
#     # annotated_frame = results[0].plot()

#     tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)

#     frame = counter.start_counting(frame, tracks)

#     return frame

#line_points =[(500, 0), (500, 1000)]
line_points = [(200, 200), (200,1080), (1000,1080), (1000,200)]  # line or region points
#line_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]  # line or region points
classes_to_count = [0,1,2,3,5,7]  # person and car classes for count




# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate=15
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)
#results = model.track(source=0,show=True,tracker="bytetra")
frame_count = 0


while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()
    frame_count += 1


    #if frame_count % 20 == 0:  # 30 frames per second * 5 seconds = 150 frames
    #frame = process_frame(frame,classes_to_count)
        

    tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)

    frame = counter.start_counting(frame, tracks)

    # Display the resulting frame
    cv2.imshow("Camera", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()
