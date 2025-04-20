# Face Recognition Based Attendance System

A modern, high-performance desktop application built with **Python**, **CustomTkinter**, and **OpenCV**. It provides an automated, real-time facial recognition solution for tracking student attendance with accurate data deduplication, thread-safe UI updates, and a light pastel UI design language.

---

## Features

- **Modern Pastel Desktop UI**: Redesigned using CustomTkinter with soft pastel color palettes, rounded cards, clean typography, and responsive grid layouts.
- **Universal Student Identity Model**: Maintains strict distinction between internal auto-incrementing serial numbers and public string Student IDs (preserving leading zeroes such as `001`).
- **Pre-Registration Validation**: Automatically blocks registration if a Student ID already exists before initializing webcam feeds or saving data.
- **Real-Time Facial Recognition**: Utilizes OpenCV Haar Cascade Classifier for detection and Local Binary Patterns Histograms (LBPH) for face recognition.
- **Duplicate Attendance Prevention**: Implements session-level and daily file-level checks ensuring students are marked only once per date.
- **Thread-Safe Live Synchronization**: Attendance records instantly sync between local CSV files and GUI table views without blocking UI execution.
- **Multi-Page Navigation**: Includes 5 dedicated navigation pages: Dashboard, Register Student, Attendance, Student Records, and Settings.

---

## Tech Stack

- **GUI Framework**: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) (Python Tkinter extension)
- **Computer Vision**: OpenCV (`opencv-contrib-python`) with Haar Cascade & LBPH Face Recognizer
- **Data Handling**: Pandas & NumPy for string extraction, standard `csv` module for storage
- **Image Processing**: Pillow (PIL)

---

## Project Structure

```text
├── Attendance/                      # Daily attendance CSV log storage
├── StudentDetails/                  # Student registration records (StudentDetails.csv)
├── TrainingImage/                   # Captured student facial training images
├── TrainingImageLabel/              # Trained LBPH model (Trainner.yml) and admin settings
├── haarcascade_frontalface_default.xml # OpenCV Haar Cascade face detection classifier
├── main.py                          # Core CustomTkinter application entry point
├── requirements.txt                 # Project python dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Documentation
```

---

## Installation & Setup

### Prerequisites

- **Python 3.9+** installed on your system.
- Webcam connected to your computer.

### Step-by-step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anjalitalan/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**:
   ```bash
   python main.py
   ```

---

## Usage Guide

1. **Register a Student**:
   - Navigate to the **Register Student** page.
   - Enter a unique **Student ID** (e.g., `001`) and **Student Name** (e.g., `John Doe`).
   - Click **Take Images**. The webcam will capture 100 sample images.
   - Click **Save Profile (Train)**, enter the admin password (default setup on first run), and train the LBPH recognition model.

2. **Track Attendance**:
   - Navigate to the **Attendance** page.
   - Click **Start Attendance Camera**.
   - The application recognizes registered faces, logs the entry to `Attendance/Attendance_<DATE>.csv`, and updates the GUI table in real time.
   - Press `Q` to close the camera feed.

3. **View Records**:
   - View student directories under **Student Records** and inspect daily logs on the **Dashboard**.

---

## Screenshots & UI Overview

| Navigation | Description |
|---|---|
| **Dashboard** | Real-time status cards, system statistics, and recent activity logs. |
| **Register Student** | Form input for Student ID/Name with progress indicator and 2-step registration. |
| **Attendance** | Prominent tracking panel with live recognition table feeds. |
| **Student Records** | Complete directory of registered students with serial numbers and IDs. |

---

## Privacy Notice

This software processes facial biometric data locally on your device. Facial images and model weights (`Trainner.yml`) are stored strictly on the local filesystem and are never transmitted over external networks or third-party cloud servers.

---

## Author

**Anjali Talan**
