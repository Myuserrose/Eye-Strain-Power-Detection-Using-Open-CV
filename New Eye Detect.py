import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import winsound
from tkinter import messagebox, Tk
import time
import os

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate eye health based on EAR
def eye_health(ear):
    if ear < 0.2:
        return "Eye strain possible: EAR too low"
    elif ear > 0.3:
        return "Possible fatigue detected: EAR too high"
    else:
        return "Normal"

# Function to generate a report
def generate_report(name, age, left_ear, right_ear, eye_condition):
    report = f"""EYE HEALTH REPORT:
    -------------------------------
    PATIENT NAME: {name}
    AGE: {age}
    LEFT EYE EAR: {left_ear:.2f}
    RIGHT EYE EAR: {right_ear:.2f}

    EYE CONDITION: {eye_condition}
    
    RECOMENDATIONS:
    -------------------------------
    - Rest your eyes regularly.
    - Try the 20-20-20 rule: Take a 20-second break every 20 minutes, looking at something 20 feet away.

     NUTRIENTS FOR EYE HEALTH :
    -------------------------------
    - Carrots (Rich in Vitamin A)
    - Spinach (Rich in lutein)
    - Eggs (Rich in Vitamin A and Zinc)
    - Fish (Omega-3 fatty acids for eye health)
    """

    # Unique doctor advice based on the condition
    if eye_condition == "Eye strain possible: EAR too low":
        report += " - Consider taking short breaks frequently to reduce strain."
    elif eye_condition == "Possible fatigue detected: EAR too high":
        report += " - Consider using blue light filters and reducing screen time."
    elif eye_condition == "Normal":
        report += " - Maintain healthy habits to keep your eyes in good condition."

    # Save the report as a text file
    report_filename = f"eye_health_report_{name}.txt"
    with open(report_filename, "w") as file:
        file.write(report)

    # Save the report as an image (optional)
    print("Report generated:", report_filename)

# Load the face detector and facial landmark predictor
shape_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# Initialize webcam
cam = cv2.VideoCapture(0)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Tkinter window for pop-up messages
root = Tk()
root.withdraw()

# Frequency and duration for beep sound
frequency = 2500  # Hz
duration = 1000   # ms

# Ask user for their name and age before starting
name = input("Enter your name: ")
age = input("Enter your age: ")

# Flag for beep sound to be triggered only once
beep_triggered = False

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Get left and right eyes landmarks
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Check eye health based on EAR
        eye_condition_left = eye_health(left_ear)
        eye_condition_right = eye_health(right_ear)
        
        # Use the more serious condition for both eyes (i.e., if either eye has a serious condition)
        if eye_condition_left != "Normal" or eye_condition_right != "Normal":
            eye_condition = "Eye strain possible" if "Eye strain" in eye_condition_left or "Eye strain" in eye_condition_right else "Possible fatigue detected"
        else:
            eye_condition = "Normal"
        
        # If condition changes, beep once
        if not beep_triggered:
            winsound.Beep(frequency, duration)
            beep_triggered = True
        
        # Display eye condition on the frame
        cv2.putText(frame, f"Eye Condition: {eye_condition}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw circles around the eyes
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    # Show the webcam frame
    cv2.imshow("Eye Health Monitor", frame)

    # Break the loop when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # Display a pop-up message with the current condition
        messagebox.showinfo("Eye Health Status", f"Current Eye Condition: {eye_condition}")
        
        # Generate a report after the pop-up message
        generate_report(name, age, left_ear, right_ear, eye_condition)
        
        break

# Clean up
cam.release()
cv2.destroyAllWindows()
