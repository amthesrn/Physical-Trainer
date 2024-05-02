import cv2 as cv
import numpy as np
import mediapipe as mp
import time

class PoseEstimator():
    def __init__(self, mode=False, complexity=1, upBody=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.upBody, self.smooth, self.detectionCon, self.trackingCon)

    def findPose(self, image, draw=True):
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return image

    def findPosition(self, image, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, land in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(land.x * w), int(land.y * h) 
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(image, (cx, cy), 3, (255, 0, 0), cv.FILLED)
        return lmList

    def findAngle(self, p1, p2, p3):
        x1, y1 = self.results.pose_landmarks.landmark[p1].x, self.results.pose_landmarks.landmark[p1].y
        x2, y2 = self.results.pose_landmarks.landmark[p2].x, self.results.pose_landmarks.landmark[p2].y
        x3, y3 = self.results.pose_landmarks.landmark[p3].x, self.results.pose_landmarks.landmark[p3].y

        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        return angle

def main():
    cap = cv.VideoCapture(0)  # 0 for the default camera, you can change it if you have multiple cameras
    detector = PoseEstimator()

    # Initialize count variables
    curl_count = 0
    curl_in_progress = False
    prev_curl_angle = 0

    # Set the desired screen size
    screen_width = 1280
    screen_height = 720

    while True:
        success, image = cap.read()
        if not success:
            print("Failed to read frame from the camera feed.")
            break

        # Resize the captured frame
        image = cv.resize(image, (screen_width, screen_height))

        image = detector.findPose(image)
        lmList = detector.findPosition(image)
        
        if len(lmList) > 15:
            curl_angle = detector.findAngle(11, 13, 15)
            print("Current curl angle:", curl_angle)

            # Draw bar for curl motion
            per = np.interp(curl_angle, (250, 350), (0, 100))
            bar = np.interp(curl_angle, (250, 350), (100, 650))
            cv.rectangle(image, (1100, 100), (1175, 650), (0, 255, 0), cv.FILLED)
            cv.rectangle(image, (1100, int(bar)), (1175, 650), (0, 255, 0), cv.FILLED)
            cv.putText(image, f'{int(per)} %', (1100, 75), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

            # Detect the down to up movement of the arm (curl)
            if curl_angle >= 250 and prev_curl_angle < 250 and not curl_in_progress:
                curl_in_progress = True
            
            if curl_angle < 250 and prev_curl_angle >= 250 and curl_in_progress:
                curl_count += 1
                curl_in_progress = False
                print("Curl detected. Count:", curl_count)
            
            prev_curl_angle = curl_angle

            # Draw curl count
            cv.rectangle(image, (0,450), (250, 720), (0, 255, 0), cv.FILLED)
            cv.putText(image, str(int(curl_count)), (45, 670), cv.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        cv.imshow("Image", image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
