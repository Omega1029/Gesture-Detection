from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

def track_video(video_path):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    output_path = "output_tracked_video.mp4"
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,
            frame_height)
    )


    while cap.isOpened():
        success, frame = cap.read()


        if success:

            results = model.track(frame, persist=True)

            boxes = results[0].boxes.xywh.cpu()
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )

            annotated_frame=results[0].plot()

            if track_ids:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)


                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))

                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=2
                    )


                out.write(annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()