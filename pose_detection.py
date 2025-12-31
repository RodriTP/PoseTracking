import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult
from landmarker_result import LandmarkerResult
import time

# --- POSE CONNECTION CONSTANTS ---
# These pairs represent the lines of the skeleton (e.g., 11-12 is shoulder to shoulder)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # Arms
    (0, 12), (1, 13), (12, 13), # Torso
    (12, 14), (13, 15), (14, 16), (15, 17) # Legs
]

# Initialize live camera feed with openCV
capture = cv2.VideoCapture(0)#TODO : change the index if you have multiple cameras
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

#initializing task for model
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
landmarkerResult = LandmarkerResult(True)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=landmarkerResult.callbackResult)

# Create a pose landmarker instance with the live stream mode:
with PoseLandmarker.create_from_options(options) as landmarker:
   
    #Start the camera feed loop
    while capture.isOpened():
        # Use time.time() for the timestamp as capture.get() can sometimes return 0 on webcams
        timestamp = int(time.time() * 1000)

        # Capture frame-by-frame
        ret, cameraFeed = capture.read()

        cameraFeed = cv2.resize(cameraFeed, (860,640))
        h,w, _ = cameraFeed.shape

        #coloring conversion for mediapipe
        feedForDetection = cv2.cvtColor(cameraFeed, cv2.COLOR_BGR2RGB)
        #Transform the OpenCV image to mediapipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=feedForDetection)  
        # Use the landmarker to detect poses in the input camera feed.
        landmarker.detect_async(mp_image, timestamp)        

        # Check if the callback has stored a result yet
        if landmarkerResult.result and landmarkerResult.result.pose_landmarks:

            # --- DRAWING SECTION ---
            for pose_landmarks in landmarkerResult.result.pose_landmarks:
                # 1. Draw the Lines (Connections)
                for connection in POSE_CONNECTIONS:
                    start_point = pose_landmarks[connection[0]]
                    end_point = pose_landmarks[connection[1]]
                    
                    # Convert normalized coordinates to pixel coordinates
                    pt1 = (int(start_point.x * w), int(start_point.y * h))
                    pt2 = (int(end_point.x * w), int(end_point.y * h))
                    cv2.line(cameraFeed, pt1, pt2, (255, 255, 255), 2) # White lines

                # 2. Draw the Dots (Landmarks)
                for landmark in pose_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(cameraFeed, (cx, cy), 4, (0, 0, 255), -1) # Red dots
        

        # Display the resulting frame
        cv2.imshow('Camera Feed', cameraFeed)

        #exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()





