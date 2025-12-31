import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult

class LandmarkerResult:
    def __init__(self, filterLandmarks : bool = False):
        self.filterLandmarks = filterLandmarks
        self.result = None
        self.detectedPose = None

    #fuction to extract the result
    def callbackResult(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if not result.pose_landmarks == None:
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