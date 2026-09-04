import cv2
import time
from ultralytics import YOLO

def run_vision_analytics():
    # Load lightweight pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Initialize video capture (0 for default webcam, or provide video file path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    prev_time = 0

    print("Starting VisionPulse analytics stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Run object detection and tracking
        results = model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()

        # Count total detected persons
        person_count = 0
        boxes = results[0].boxes
        if boxes is not None and boxes.cls is not None:
            for cls_id in boxes.cls:
                if int(cls_id) == 0:  # Class 0 is 'person' in COCO dataset
                    person_count += 1

        # Overlay analytics dashboard on the frame
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Active Persons: {person_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 100, 0),
            2
        )

        # Display the live output
        cv2.imshow("VisionPulse - Real-time Analytics", annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_analytics()
