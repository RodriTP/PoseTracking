import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult

UNFILTERED_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24), # Torso
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 31), (28, 32) # Legs
]

FILTERED_POSE_CONNECTIONS = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # Arms
    (0, 12), (1, 13), (12, 13), # Torso
    (12, 14), (13, 15), (14, 16), (15, 17) # Legs
]


class LandmarkerResult:
    
    def __init__(self, filterLandmarks : bool = False):
        self.filterLandmarks = filterLandmarks
        self.result = None
        self.detectedPose = None

    #fuction to extract the result
    def callbackResult(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result and result.pose_landmarks:
            if self.filterLandmarks:
                self.result = result
                self.__removeLandmarkers__()
            else:
                self.result = result
            #print(f'Pose landmarks: {result.pose_landmarks}\n')
            print(len(result.pose_landmarks[0]))
        else:
            self.result = None

    
    def __removeLandmarkers__(self):
        """
        Remove landmarks from the face and feet from the result
        """
        if not self.result == None:
            for i in range(11):#removing face landmarks
                self.result.pose_landmarks[0].pop(0)
            for i in range(4): #removing feet landmarks
                self.result.pose_landmarks[0].pop() 

    def drawResult(self, image, height:int, width:int):
        """
        Draw the landmarks on the image
        """

        if self.result and self.result.pose_landmarks:
            if self.filterLandmarks:
                connections = FILTERED_POSE_CONNECTIONS
            else:
                connections = UNFILTERED_POSE_CONNECTIONS

            for pose_landmarks in self.result.pose_landmarks:
                # 1. Draw the Lines (Connections)
                for connection in connections:
                    start_point = pose_landmarks[connection[0]]
                    end_point = pose_landmarks[connection[1]]

                    # Convert normalized coordinates to pixel coordinates
                    pt1 = (int(start_point.x * width), int(start_point.y * height))
                    pt2 = (int(end_point.x * width), int(end_point.y * height))
                    cv2.line(image, pt1, pt2, (255, 255, 255), 2) # White lines
                # 2. Draw the Dots (Landmarks)
                for landmark in pose_landmarks:
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1) # Red dots