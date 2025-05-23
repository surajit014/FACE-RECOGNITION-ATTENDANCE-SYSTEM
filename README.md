# Face Recognition Attendance System

A web-based attendance system that uses face recognition to automatically track employee/student attendance. The system captures images through a web interface, recognizes faces using machine learning models, and records entry and exit times in a MySQL database.

## Features

- Real-time face detection and recognition
- Automatic attendance tracking with entry and exit timestamps
- Web-based interface for easy access
- Support for multiple users
- Secure database storage
- Responsive design for various devices

## Technical Stack

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript
- **Face Recognition**: 
  - MTCNN for face detection
  - FaceNet (InceptionResnetV1) for face embedding
  - Random Forest Classifier for face recognition
- **Database**: MySQL
- **Dependencies**: 
  - facenet-pytorch
  - torch
  - PIL
  - numpy
  - mysql-connector-python

## Prerequisites

- Python 3.11.2
- MySQL Server
- Webcam-enabled device
- Required Python packages (install via pip):
  ```
  flask
  facenet-pytorch
  torch
  pillow
  numpy
  mysql-connector-python
  joblib
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd Attendance
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up MySQL database:
   - Create a database named 'attendance'
   - Create a table with the following structure:
     ```sql
     CREATE TABLE attendance (
         id INT AUTO_INCREMENT PRIMARY KEY,
         name VARCHAR(255),
         entry_time DATETIME,
         exit_time DATETIME
     );
     ```

4. Configure database connection:
   - Update the database credentials in `app.py`:
     ```python
     host="localhost"
     user="user_name"
     password="Database_Password"
     database="Database_name"
     ```

5. Place your trained models:
   - Ensure `face_rf_model.pkl` and `label_encoder.pkl` are in the project root directory

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Allow camera access when prompted

4. The system will:
   - Capture and detect faces
   - Recognize the person
   - Allow marking entry/exit based on attendance status
   - Store the attendance record in the database

## Project Structure

```
Attendance/
├── app.py              # Main Flask application
├── face_rf_model.pkl   # Trained face recognition model
├── label_encoder.pkl   # Label encoder for face recognition
├── static/            # Static files (images, CSS, JS)
├── templates/         # HTML templates
│   ├── index.html    # Main page
│   └── result.html   # Results page
└── train_model.py    # Script for training the face recognition model
```

## Security Considerations

- The system uses session management for secure operations
- Database credentials should be properly secured
- Consider implementing additional authentication for admin access
- Regular backup of the attendance database is recommended

