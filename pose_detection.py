import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Class to excract the results from the async callback
class LandmarkerResult:
    def __init__(self):
        self.result = None

    #fuction to callback to print results
    def print_result(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result
#    print(f'pose landmarker result: {result}')
    

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
result = LandmarkerResult()

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result.print_result)

# Create a pose landmarker instance with the live stream mode:
with PoseLandmarker.create_from_options(options) as landmarker:
   
    #Start the camera feed loop
    while capture.isOpened():
        frame_timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))

        # Capture frame-by-frame
        ret, cameraFeed = capture.read()

        #coloring conversion for mediapipe
        feedForDetection = cv2.cvtColor(cameraFeed, cv2.COLOR_BGR2RGB)

        #Transform the OpenCV image to mediapipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=feedForDetection)  
        # Use the landmarker to detect poses in the input camera feed.
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        print(result.result)
        
        #Change the color back to BGR for OpenCV
        cameraFeed = cv2.cvtColor(feedForDetection, cv2.COLOR_RGB2BGR)
        #mp_drawing.draw_landmarks(cameraFeed, detectionResult.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the resulting frame
        cv2.imshow('Camera Feed', cameraFeed)

        #exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()





