import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables to store previous landmarks
prev_landmarks = None

# Video feed
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('input_video.mp4')

# Setup mediapipe instance
with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Detect objects and render
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make Detection
        results = pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Joint extraction
        try:
            landmarks = results.pose_landmarks.landmark
            if prev_landmarks is not None:
                # Example: Calculate the movement of the RIGHT ELBOW landmark
                current_right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                previous_right_elbow_y = prev_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y

                # Example: Calculate the movement of the LEFT ELBOW landmark
                current_left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
                previous_left_elbow_y = prev_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y

                # Define movement thresholds (you can adjust these)
                movement_threshold = 0.08  # Adjust based on your needs

                # Label movements based on landmarks' positions
                if current_right_elbow_y < previous_right_elbow_y - movement_threshold:
                    print("Raise Right Hand")

                if current_left_elbow_y < previous_left_elbow_y - movement_threshold:
                    print("Raise Left Hand")

                #Nose
                mt= 0.01
                current_nose_x = landmarks[mp_pose.PoseLandmark.NOSE].x
                previous_nose_x = prev_landmarks[mp_pose.PoseLandmark.NOSE].x
                movement = current_nose_x - previous_nose_x

                
                if abs(movement) > mt and not notification_sent:
                    print("Movement of NOSE detected!")
                    notification_sent = True  # Set the notification flag to True

                # Reset the notification flag if the nose movement is small
                if abs(movement) <= mt:
                    notification_sent = False
              
            prev_landmarks = landmarks  # Update previous landmarks
        except:
            pass

        # Render Detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
