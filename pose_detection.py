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
landmarkerResult = LandmarkerResult(False)

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
        
        landmarkerResult.drawResult(cameraFeed, h, w)
        # Display the resulting frame
        cv2.imshow('Camera Feed', cameraFeed)

        #exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()





