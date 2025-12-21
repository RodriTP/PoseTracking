import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import time


#Class to excract the results from the async callback
class LandmarkerResult:
    def __init__(self):
        self.result = None

    #fuction to callback to print results
    def print_result(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result
        print(f'Pose landmarks: {result.pose_landmarks}')

# --- POSE CONNECTION CONSTANTS ---
# These pairs represent the lines of the skeleton (e.g., 11-12 is shoulder to shoulder)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 31), (28, 32) # Legs
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
landmarker_data = LandmarkerResult()

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=landmarker_data.print_result)

# Create a pose landmarker instance with the live stream mode:
with PoseLandmarker.create_from_options(options) as landmarker:
   
    #Start the camera feed loop
    while capture.isOpened():
        # Use time.time() for the timestamp as capture.get() can sometimes return 0 on webcams
        timestamp = int(time.time() * 1000)

        # Capture frame-by-frame
        ret, cameraFeed = capture.read()

        h,w, _ = cameraFeed.shape

        #coloring conversion for mediapipe
        feedForDetection = cv2.cvtColor(cameraFeed, cv2.COLOR_BGR2RGB)
        #Transform the OpenCV image to mediapipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=feedForDetection)  
        # Use the landmarker to detect poses in the input camera feed.
        landmarker.detect_async(mp_image, timestamp)
        
        # --- DRAWING SECTION ---
        # Check if the callback has stored a result yet
        if landmarker_data.result and landmarker_data.result.pose_landmarks:
            for pose_landmarks in landmarker_data.result.pose_landmarks:
                
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





