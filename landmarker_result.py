import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult

# These pairs represent the lines of the skeleton (e.g., 11-12 is shoulder to shoulder)
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
        self.result : mp.tasks.vision.PoseLandmarkerResult = None
        self.detectedPose = None
        self.center_hip_point : tuple = None
        self.neck_point : tuple = None

    #fuction to extract the result
    def callbackResult(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result and result.pose_landmarks:
            if self.filterLandmarks:
                self.result = result
                self.__removeLandmarkers__()
            else:
                self.result = result

            #print(f'Pose landmarks: {result.pose_landmarks}\n')
            #print(len(result.pose_landmarks[0]))
        else:
            self.result = None

    
    def __removeLandmarkers__(self):
        """
        Remove landmarks from the face and feet from the result
        """
        if not self.result == None:
            for i in range(11):#removing face landmarks
                self.result.pose_landmarks[0].pop(0)
                self.result.pose_world_landmarks[0].pop(0)
            for i in range(4): #removing feet landmarks
                self.result.pose_landmarks[0].pop()
                self.result.pose_world_landmarks[0].pop()

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

                    # Draw lines with different colors based on detected pose
                    if self.filterLandmarks and self.detectedPose == 'T Pose':
                        if connection == (0, 1) or connection == (0, 2) or connection == (1, 3) or connection == (2, 4) or connection == (3, 5):
                            cv2.line(image, pt1, pt2, (0, 255, 0), 2) # Green lines
                        else:
                            cv2.line(image, pt1, pt2, (255, 255, 255), 2) # White lines

                    else:
                        cv2.line(image, pt1, pt2, (255, 255, 255), 2) # White lines
                # 2. Draw the Dots (Landmarks)
                for landmark in pose_landmarks:
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1) # Red dots
                
                if self.neck_point and self.center_hip_point and self.detectedPose == 'T Pose':
                    start_point = pose_landmarks[connection[0]]
                    end_point = pose_landmarks[connection[1]]

                    pt1 = (int((pose_landmarks[0].x + pose_landmarks[1].x) * width), 
                           int((pose_landmarks[0].y + pose_landmarks[1].y) * height))
                    pt2 = (int((pose_landmarks[12].x + pose_landmarks[13].x) * width),
                            int((pose_landmarks[12].y + pose_landmarks[13].y) * height))

                    cv2.line(image, pt1, pt2, (0, 255, 0), 2) # Green line for torso
                    cv2.circle(image, (self.neck_point[0], self.neck_point[1]), 4, (0, 0, 255), -1) # Red dots
                    cv2.circle(image, (self.center_hip_point[0], self.center_hip_point[1]), 4, (0, 0, 255), -1) # Red dots
        
    def isPoseDetected(self):
        """
        Returns true if a pose is detected, the detected pose is in self.detectedPose\n
        This methode use the world landmarks to determine the pose\n
        Valid poses are T pose, U pose and N pose
        """
        if self.result and self.result.pose_world_landmarks:
            #variables for analysis
            n : float = 0.0
            x_sum : float = 0.0 
            y_sum : float = 0.0
            xy_sum : float = 0.0
            x2_sum : float = 0.0

            neck_point = (self.result.pose_world_landmarks[0][0].x+ self.result.pose_world_landmarks[0][1].x,
                          self.result.pose_world_landmarks[0][0].y + self.result.pose_world_landmarks[0][1].y,
                          self.result.pose_world_landmarks[0][0].z + self.result.pose_world_landmarks[0][1].z)
            center_hip_point = (self.result.pose_world_landmarks[0][12].x+self.result.pose_world_landmarks[0][13].x,
                                self.result.pose_world_landmarks[0][12].y + self.result.pose_world_landmarks[0][13].y,
                               self.result.pose_world_landmarks[0][12].z + self.result.pose_world_landmarks[0][13].z)
            
            #T pose detection
            n = 6
            for i in range(6):
                x_sum += self.result.pose_world_landmarks[0][i].x
                y_sum += self.result.pose_world_landmarks[0][i].y
                xy_sum += self.result.pose_world_landmarks[0][i].x * self.result.pose_world_landmarks[0][i].y
                x2_sum += self.result.pose_world_landmarks[0][i].x * self.result.pose_world_landmarks[0][i].x
            
            arms_slope = (n*xy_sum-x_sum*y_sum)/(n*x2_sum - x_sum*x_sum)
            torso_slope = (center_hip_point[1]-neck_point[1])/(center_hip_point[0]-neck_point[0])
            
            #print(f'Arms slope: {arms_slope}, Torso slope: {torso_slope}')
            #print(f'inverse of torso slope: {-1/torso_slope}')
            #print(f'diffrence between slopes: {abs(abs(arms_slope) + (-1/abs(torso_slope)))}\n')


            if abs(arms_slope) < 0.05 and abs(abs(arms_slope) + (-1/abs(torso_slope))) < 0.05 \
            and abs(abs(self.result.pose_world_landmarks[0][3].y)-abs(self.result.pose_world_landmarks[0][1].y)) < 0.1 \
            and abs(abs(self.result.pose_world_landmarks[0][5].y)-abs(self.result.pose_world_landmarks[0][1].y)) < 0.1 \
            and abs(abs(self.result.pose_world_landmarks[0][2].y)-abs(self.result.pose_world_landmarks[0][0].y)) < 0.1 \
            and abs(abs(self.result.pose_world_landmarks[0][4].y)-abs(self.result.pose_world_landmarks[0][0].y)) < 0.1:
                self.detectedPose = 'T Pose'
                return True
            
            #print(len(self.result.pose_world_landmarks[0]))
            print(f'World landmarks: {self.result.pose_world_landmarks[0]}\n')

        self.detectedPose = 'No Pose Detected'
        return False