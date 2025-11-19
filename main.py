import cv2
import numpy as np
import face_recognition
import os
import csv
import threading
import webbrowser
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_file, request  # Added "request"

app = Flask(__name__, static_folder="static")

STUDENT_FOLDER = "students/"
ATTENDANCE_FILE = "attendance.csv"

known_face_encodings = []
known_face_names = []

# Ensure students folder exists
if not os.path.exists(STUDENT_FOLDER):
    os.makedirs(STUDENT_FOLDER)

# Load images from the students folder into memory
for file in os.listdir(STUDENT_FOLDER):
    if file.endswith((".jpg", ".png")):
        img_path = os.path.join(STUDENT_FOLDER, file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(file)[0])

# Create the CSV with a header if it does not exist
if not os.path.isfile(ATTENDANCE_FILE):
    print(f"[INFO] Creating new CSV file {ATTENDANCE_FILE} with header.")
    with open(ATTENDANCE_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Student Name", "Status"])

marked_today = set()  # Tracks who is already marked present today
last_checked_date = None

# Tracks whether a face is actively in frame vs. out of frame
# We mark "IN_FRAME_FIRST_TIME" when a new person is recognized for the day
# Once they leave the frame, their status is changed to "LEFT_FRAME"
face_state = {}


def load_marked_today_from_csv():
    """
    Read today's attendance from ATTENDANCE_FILE and populate the
    'marked_today' set with all names marked 'Present' for the current date.
    """
    global marked_today
    marked_today.clear()
    today_date = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if len(row) >= 4:
                    date, time_log, student_name, status = row
                    if date == today_date and status == "Present":
                        marked_today.add(student_name)


def reset_daily_attendance():
    """
    If the date has changed since last check, clear today's attendance
    and face states for the new day, then reload marked_today from CSV.
    """
    global last_checked_date, marked_today, face_state
    today_date = datetime.now().strftime("%Y-%m-%d")

    if last_checked_date != today_date:
        print(f"[INFO] New day detected. Resetting daily attendance for {today_date}.")
        marked_today.clear()
        face_state.clear()
        last_checked_date = today_date
        load_marked_today_from_csv()


def is_already_present_today(name):
    """
    Check if a name has already been recorded as present for today.
    """
    return name in marked_today


def mark_attendance(name):
    """
    Append a new entry to the CSV file for a newly discovered attendee.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"[INFO] Marking {name} as present at {current_time} on {today_date}.")
    with open(ATTENDANCE_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([today_date, current_time, name, "Present"])

    marked_today.add(name)


def generate_frames():
    """
    Continuously capture frames from the camera, detect/recognize faces,
    and label them on the video stream, also marking attendance in CSV.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    # Continuously read frames from the webcam
    while True:
        reset_daily_attendance()  # Check date/time once each loop
        success, frame = cap.read()
        if not success:
            break

        # Convert from BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        faces_found_this_frame = set()
        labels_for_frame = []  # (bbox, label)

        # Compare each detected face with known faces
        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.5)
            label = "Not Available"

            if True in matches:
                matched_indices = [i for i, match in enumerate(matches) if match]
                matched_names = [known_face_names[i] for i in matched_indices]
                if matched_names:
                    recognized_name = matched_names[0]
                    faces_found_this_frame.add(recognized_name)

                    if not is_already_present_today(recognized_name):
                        # First time detection today -> mark as present
                        mark_attendance(recognized_name)
                        face_state[recognized_name] = "IN_FRAME_FIRST_TIME"
                        display_label = f"Present: {recognized_name}"
                    else:
                        # If already present, see if user is in frame for the first time or returning
                        current_state = face_state.get(recognized_name)
                        if current_state == "IN_FRAME_FIRST_TIME":
                            # Still in frame from first detection
                            display_label = f"Present: {recognized_name}"
                        else:
                            # They left frame once, now returning
                            display_label = f"Already Present: {recognized_name}"
                else:
                    display_label = label
            else:
                display_label = label

            labels_for_frame.append(((top, right, bottom, left), display_label))

        # Update states for people who were previously in frame but not found this time
        for recorded_name, current_state in list(face_state.items()):
            if current_state == "IN_FRAME_FIRST_TIME" and recorded_name not in faces_found_this_frame:
                face_state[recorded_name] = "LEFT_FRAME"

        # Draw bounding boxes and labels on the frame
        for (top, right, bottom, left), text_label in labels_for_frame:
            if text_label.startswith("Present"):
                color = (0, 255, 0)       # Green
            elif text_label.startswith("Already Present"):
                color = (0, 165, 255)    # Orange
            elif text_label == "Not Available":
                color = (0, 0, 255)      # Red
            else:
                color = (255, 255, 255)  # White

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text_label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Encode the augmented frame into JPEG to stream over HTTP
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame in a byte format for the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """
    Renders the main page. Make sure there's an 'index.html' inside 'templates/' folder.
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Route that streams video frames back to the client.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/download_csv', methods=['GET'])
def download_csv():
    """
    Endpoint to download the CSV file, now password-protected.
    """
    REQUIRED_PASSWORD = "admin@123"  # Example password; store securely in production

    user_password = request.args.get("pwd")
    if not user_password:
        return jsonify({"error": "No password provided."}), 400

    if user_password == REQUIRED_PASSWORD:
        if os.path.exists(ATTENDANCE_FILE):
            return send_file(ATTENDANCE_FILE, as_attachment=True)
        return jsonify({"message": "No attendance records available!"}), 404
    else:
        return jsonify({"error": "Incorrect password."}), 401


@app.route('/attendance_stats')
def attendance_stats():
    """
    Returns total students and how many are marked present today.
    Useful if you want to display stats on a dashboard.
    """
    total_students = len(known_face_names)
    present_today = len(marked_today)
    return jsonify({"total_students": total_students, "present_today": present_today})


@app.route('/extended_attendance')
def extended_attendance():
    total_students = len(known_face_names)
    present = len(marked_today)
    absent = total_students - present
    percentage = (present / total_students * 100) if total_students > 0 else 0
    return jsonify({
        "total_students": total_students,
        "present": present,
        "absent": absent,
        "percentage": percentage
    })


def open_browser():
    """
    Helper function to open the default web browser to your app's local URL.
    """
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == '__main__':
    # Open the browser once, after a 1.5 second delay
    threading.Timer(1.5, open_browser).start()
    # Run Flask without debug auto-reloader so it only opens one browser tab
    app.run(debug=False)
