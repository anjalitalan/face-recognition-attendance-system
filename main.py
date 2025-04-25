"""
Face Recognition Based Attendance System
----------------------------------------
A modern CustomTkinter desktop application utilizing OpenCV Haar Cascade Classifier
and LBPH (Local Binary Patterns Histograms) Face Recognizer for real-time facial recognition attendance tracking.

Author: Anjali Talan
"""

import os
import cv2
import csv
import time
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import customtkinter as ctk

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

COLOR_MAIN_BG = "#F7F4EF"
COLOR_SIDEBAR_BG = "#EFE9E2"
COLOR_PANEL_BG = "#F1EBE5"
COLOR_CARD_PINK = "#F4BFC1"
COLOR_CARD_PEACH = "#F8D8AD"
COLOR_CARD_LAVENDER = "#CBC7F6"
COLOR_CARD_MINT = "#BDEBD5"
COLOR_WHITE = "#FFFDFC"
COLOR_TEXT_MAIN = "#171717"
COLOR_TEXT_MUTED = "#6F6A65"
COLOR_BTN_DARK = "#111111"
COLOR_BTN_DARK_HOVER = "#2A2A2A"

def assure_path_exists(path):
    """ Ensures directory exists on local disk """
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

def normalize_student_id(value):
    """
    Normalizes Student ID values as clean strings while preserving leading zeroes (e.g., '001').
    Trims trailing '.0' if parsed as float strings.
    """
    if value is None:
        return ""
    val = str(value).strip()
    if val.endswith('.0'):
        val = val[:-2]
    return val

def clean_name_str(val):
    """ Cleans raw pandas Series or list representations into pure plain text """
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) > 0:
            val = str(val[0])
        else:
            val = "Unknown"
    val_str = str(val).strip()
    val_str = val_str.replace("['", "").replace("']", "").replace('["', '').replace('"]', '').replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
    return val_str if val_str else "Unknown"

def check_haarcascadefile():
    """ Verifies presence of Haar Cascade xml file for face detection """
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if not exists:
        mess.showerror(title='File Missing', message='haarcascade_frontalface_default.xml is missing!\nPlease contact support.')
        return False
    return True

def load_and_clean_student_details():
    """
    Reads, cleans, and deduplicates StudentDetails/StudentDetails.csv.
    Creates a backup StudentDetails_backup.csv before deduplication.
    Rule: Keep first valid occurrence of each Student ID, drop later duplicates.
    Returns: List of dicts [{"serial_no": int, "student_id": str, "name": str}, ...]
    """
    csv_path = os.path.join("StudentDetails", "StudentDetails.csv")
    backup_path = os.path.join("StudentDetails", "StudentDetails_backup.csv")

    if not os.path.isfile(csv_path):
        return []

    records = []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as csvFile:
        reader = csv.reader(csvFile)
        for lines in reader:
            non_empty = [c.strip() for c in lines if c.strip()]
            if not non_empty or non_empty[0] == "SERIAL NO.":
                continue
            if len(non_empty) >= 3:
                try:
                    s_no = int(non_empty[0])
                    s_id = normalize_student_id(non_empty[1])
                    s_name = clean_name_str(non_empty[2])
                    if s_id and s_id != "ID":
                        records.append({
                            "serial_no": s_no,
                            "student_id": s_id,
                            "name": s_name
                        })
                except ValueError:
                    continue

    if not records:
        return []

    seen_ids = set()
    unique_records = []
    has_duplicates = False

    for rec in records:
        sid = rec["student_id"]
        if sid in seen_ids:
            has_duplicates = True
            print(f"[Warning] Duplicate Student ID '{sid}' ({rec['name']}) found. Keeping first record.")
        else:
            seen_ids.add(sid)
            unique_records.append(rec)

    try:
        if not os.path.isfile(backup_path):
            with open(backup_path, 'w', newline='', encoding='utf-8') as b_file:
                b_writer = csv.writer(b_file)
                b_writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
                for rec in records:
                    b_writer.writerow([rec["serial_no"], '', rec["student_id"], '', rec["name"]])

        if has_duplicates:
            with open(csv_path, 'w', newline='', encoding='utf-8') as c_file:
                c_writer = csv.writer(c_file)
                c_writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
                for rec in unique_records:
                    c_writer.writerow([rec["serial_no"], '', rec["student_id"], '', rec["name"]])
    except Exception as e:
        print(f"[Error] Failed to process backup/clean CSV: {e}")

    unique_records.sort(key=lambda x: x["serial_no"])
    return unique_records

def get_next_serial_number(records=None):
    """ Auto-generates next serial number: highest existing serial number + 1 """
    if records is None:
        records = load_and_clean_student_details()
    if not records:
        return 1
    return max(rec["serial_no"] for rec in records) + 1

def get_student_by_serial(serial_number, records=None):
    """ Canonical lookup: maps recognized internal face label (serial_no) to student record """
    if records is None:
        records = load_and_clean_student_details()
    try:
        target_s = int(serial_number)
    except (ValueError, TypeError):
        return None

    for rec in records:
        if rec["serial_no"] == target_s:
            return rec
    return None

def student_id_exists(student_id, records=None):
    """ Reusable validation function checking if normalized Student ID already exists """
    target_id = normalize_student_id(student_id)
    if not target_id:
        return False
    if records is None:
        records = load_and_clean_student_details()
    for rec in records:
        if rec["student_id"] == target_id:
            return True
    return False

def getImagesAndLabels(path):
    """ Loads face images and numeric label IDs from TrainingImage folder """
    if not os.path.exists(path):
        return [], []

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.isspace()]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            filename = os.path.split(imagePath)[-1]
            ID = int(filename.split(".")[1])
            faces.append(imageNp)
            Ids.append(ID)
        except Exception:
            continue

    return faces, Ids


class AttendanceApp(ctk.CTk):
    """ Main Application Class for Face Recognition Attendance System """

    def __init__(self):
        super().__init__()

        self.geometry("1300x800")
        self.title("Face Recognition Attendance System")
        self.configure(fg_color=COLOR_MAIN_BG)
        self.minsize(1150, 720)

        self.grid_columnconfigure(0, weight=0, minsize=220)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=300)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_buttons = {}
        self.pages_dict = {}

        self.build_sidebar()
        self.build_main_content()
        self.build_right_panel()
        self.setup_styles()

        self.show_page("Dashboard")
        self.tick()
        self.update_all_counts()
        self.update_model_status()
        self.refresh_attendance_table()
        self.load_student_records()

    def setup_styles(self):
        """ Configures Treeview table styling for consistent light aesthetic """
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=COLOR_WHITE,
            foreground=COLOR_TEXT_MAIN,
            rowheight=32,
            fieldbackground=COLOR_WHITE,
            font=('Segoe UI', 10),
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background=COLOR_SIDEBAR_BG,
            foreground=COLOR_TEXT_MAIN,
            font=('Segoe UI', 11, 'bold'),
            relief='flat'
        )
        style.map("Treeview", background=[('selected', COLOR_CARD_MINT)], foreground=[('selected', COLOR_TEXT_MAIN)])

    def build_sidebar(self):
        """ Builds navigation sidebar on the left panel """
        sidebar_frame = ctk.CTkFrame(self, fg_color=COLOR_SIDEBAR_BG, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        logo_container = ctk.CTkFrame(sidebar_frame, fg_color="transparent")
        logo_container.pack(anchor="w", padx=20, pady=(25, 30))

        logo_icon = ctk.CTkFrame(logo_container, fg_color=COLOR_BTN_DARK, corner_radius=14, width=44, height=44)
        logo_icon.pack(side="left")
        logo_icon.pack_propagate(False)
        ctk.CTkLabel(logo_icon, text="FR", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_WHITE).pack(expand=True)

        ctk.CTkLabel(logo_container, text=" Attendance\n System", font=ctk.CTkFont(size=15, weight="bold"), text_color=COLOR_TEXT_MAIN, justify="left").pack(side="left", padx=10)

        nav_items = [
            ("Dashboard", "Dashboard"),
            ("Register Student", "Register Student"),
            ("Attendance", "Attendance"),
            ("Student Records", "Student Records"),
            ("Settings", "Settings"),
        ]

        for display_name, page_key in nav_items:
            btn = ctk.CTkButton(
                sidebar_frame,
                text=f"  {display_name}",
                anchor="w",
                font=ctk.CTkFont(size=14, weight="normal"),
                fg_color="transparent",
                hover_color="#E2DCD4",
                text_color=COLOR_TEXT_MAIN,
                corner_radius=16,
                height=42,
                command=lambda p=page_key: self.show_page(p)
            )
            btn.pack(fill="x", padx=15, pady=5)
            self.sidebar_buttons[page_key] = btn

        exit_btn = ctk.CTkButton(
            sidebar_frame,
            text="Exit System",
            anchor="w",
            font=ctk.CTkFont(size=14, weight="normal"),
            fg_color="transparent",
            hover_color="#F4BFC1",
            text_color="#DC2626",
            corner_radius=16,
            height=42,
            command=self.destroy
        )
        exit_btn.pack(side="bottom", fill="x", padx=15, pady=25)

    def show_page(self, page_name):
        """ Switches active view between navigation pages """
        for name, frame in self.pages_dict.items():
            if name == page_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

        for name, btn in self.sidebar_buttons.items():
            if name == page_name:
                btn.configure(fg_color=COLOR_BTN_DARK, text_color=COLOR_WHITE)
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_TEXT_MAIN)

        if page_name == "Attendance":
            self.refresh_attendance_table()
        elif page_name == "Student Records":
            self.load_student_records()

    def build_main_content(self):
        """ Constructs central workspace pages """
        content_container = ctk.CTkFrame(self, fg_color="transparent")
        content_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        content_container.grid_rowconfigure(0, weight=1)
        content_container.grid_columnconfigure(0, weight=1)

        # PAGE 1: DASHBOARD
        page_dash = ctk.CTkFrame(content_container, fg_color="transparent")
        self.pages_dict["Dashboard"] = page_dash

        ctk.CTkLabel(page_dash, text="Dashboard Overview", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(page_dash, text="System overview, live status metrics, and quick actions", font=ctk.CTkFont(size=14), text_color=COLOR_TEXT_MUTED).pack(anchor="w", pady=(0, 20))

        dash_stats_grid = ctk.CTkFrame(page_dash, fg_color="transparent")
        dash_stats_grid.pack(fill="x", pady=(0, 20))
        for c in range(4):
            dash_stats_grid.columnconfigure(c, weight=1)

        card_s1 = ctk.CTkFrame(dash_stats_grid, fg_color=COLOR_CARD_PEACH, corner_radius=22)
        card_s1.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkLabel(card_s1, text="Total Students", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=15, pady=(12, 0))
        self.lbl_dash_students = ctk.CTkLabel(card_s1, text="0", font=ctk.CTkFont(size=26, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_dash_students.pack(anchor="w", padx=15, pady=(0, 10))

        card_s2 = ctk.CTkFrame(dash_stats_grid, fg_color=COLOR_CARD_MINT, corner_radius=22)
        card_s2.grid(row=0, column=1, padx=4, sticky="ew")
        ctk.CTkLabel(card_s2, text="Today's Attendance", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=15, pady=(12, 0))
        self.lbl_dash_today = ctk.CTkLabel(card_s2, text="0 Today", font=ctk.CTkFont(size=22, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_dash_today.pack(anchor="w", padx=15, pady=(0, 10))

        card_s3 = ctk.CTkFrame(dash_stats_grid, fg_color=COLOR_CARD_LAVENDER, corner_radius=22)
        card_s3.grid(row=0, column=2, padx=4, sticky="ew")
        ctk.CTkLabel(card_s3, text="Model Status", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=15, pady=(12, 0))
        self.lbl_dash_model = ctk.CTkLabel(card_s3, text="Checking...", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_dash_model.pack(anchor="w", padx=15, pady=(4, 12))

        card_s4 = ctk.CTkFrame(dash_stats_grid, fg_color=COLOR_CARD_PINK, corner_radius=22)
        card_s4.grid(row=0, column=3, padx=(6, 0), sticky="ew")
        ctk.CTkLabel(card_s4, text="Camera Status", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=15, pady=(12, 0))
        self.lbl_dash_camera = ctk.CTkLabel(card_s4, text="Standby", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_dash_camera.pack(anchor="w", padx=15, pady=(4, 12))

        dash_preview_card = ctk.CTkFrame(page_dash, fg_color=COLOR_WHITE, corner_radius=22)
        dash_preview_card.pack(fill="both", expand=True, pady=(0, 15))

        ctk.CTkLabel(dash_preview_card, text="Recent Attendance Activity", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 8))

        dash_tbl_frame = ctk.CTkFrame(dash_preview_card, fg_color="transparent")
        dash_tbl_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.dash_preview_tree = ttk.Treeview(dash_tbl_frame, columns=('id', 'name', 'date', 'time'), show='headings', selectmode='browse', height=6)
        self.dash_preview_tree.heading('id', text='STUDENT ID')
        self.dash_preview_tree.heading('name', text='STUDENT NAME')
        self.dash_preview_tree.heading('date', text='DATE')
        self.dash_preview_tree.heading('time', text='LOG TIME')

        self.dash_preview_tree.column('id', width=100, anchor='center')
        self.dash_preview_tree.column('name', width=180, anchor='w')
        self.dash_preview_tree.column('date', width=110, anchor='center')
        self.dash_preview_tree.column('time', width=110, anchor='center')
        self.dash_preview_tree.pack(fill="both", expand=True)

        dash_actions = ctk.CTkFrame(page_dash, fg_color=COLOR_SIDEBAR_BG, corner_radius=18)
        dash_actions.pack(fill="x")
        ctk.CTkLabel(dash_actions, text="Quick Shortcuts:", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(side="left", padx=20, pady=12)

        ctk.CTkButton(dash_actions, text="Open Attendance", fg_color=COLOR_BTN_DARK, hover_color=COLOR_BTN_DARK_HOVER, text_color=COLOR_WHITE, corner_radius=14, height=36, command=lambda: self.show_page("Attendance")).pack(side="left", padx=5)
        ctk.CTkButton(dash_actions, text="Register New Student", fg_color=COLOR_WHITE, hover_color="#F3EFEA", text_color=COLOR_TEXT_MAIN, corner_radius=14, height=36, command=lambda: self.show_page("Register Student")).pack(side="left", padx=5)
        ctk.CTkButton(dash_actions, text="View Records", fg_color=COLOR_WHITE, hover_color="#F3EFEA", text_color=COLOR_TEXT_MAIN, corner_radius=14, height=36, command=lambda: self.show_page("Student Records")).pack(side="left", padx=5)

        # PAGE 2: REGISTER STUDENT
        page_reg = ctk.CTkFrame(content_container, fg_color="transparent")
        self.pages_dict["Register Student"] = page_reg

        ctk.CTkLabel(page_reg, text="Register Student", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(page_reg, text="Capture face images and train recognition profile", font=ctk.CTkFont(size=14), text_color=COLOR_TEXT_MUTED).pack(anchor="w", pady=(0, 20))

        reg_card = ctk.CTkFrame(page_reg, fg_color=COLOR_CARD_LAVENDER, corner_radius=26)
        reg_card.pack(fill="both", expand=True)

        reg_form_inner = ctk.CTkFrame(reg_card, fg_color=COLOR_WHITE, corner_radius=22)
        reg_form_inner.pack(padx=30, pady=25, fill="x")

        lbl_id = ctk.CTkLabel(reg_form_inner, text="Student ID", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_id.grid(row=0, column=0, sticky="w", padx=20, pady=(15, 2))

        self.txt_id = ctk.CTkEntry(reg_form_inner, placeholder_text="Enter Unique ID (e.g. 001)", width=260, height=38, corner_radius=12, fg_color=COLOR_MAIN_BG, border_color="#E2DCD4", text_color=COLOR_TEXT_MAIN)
        self.txt_id.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 15))

        btn_clear_id = ctk.CTkButton(reg_form_inner, text="Clear ID", fg_color=COLOR_SIDEBAR_BG, hover_color="#E2DCD4", text_color=COLOR_TEXT_MAIN, corner_radius=12, width=90, height=38, command=self.clear)
        btn_clear_id.grid(row=1, column=1, padx=(10, 20), pady=(0, 15))

        lbl_name = ctk.CTkLabel(reg_form_inner, text="Student Name", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_name.grid(row=2, column=0, sticky="w", padx=20, pady=(5, 2))

        self.txt_name = ctk.CTkEntry(reg_form_inner, placeholder_text="Enter Full Name (e.g. John Doe)", width=260, height=38, corner_radius=12, fg_color=COLOR_MAIN_BG, border_color="#E2DCD4", text_color=COLOR_TEXT_MAIN)
        self.txt_name.grid(row=3, column=0, sticky="w", padx=20, pady=(0, 20))

        btn_clear_name = ctk.CTkButton(reg_form_inner, text="Clear Name", fg_color=COLOR_SIDEBAR_BG, hover_color="#E2DCD4", text_color=COLOR_TEXT_MAIN, corner_radius=12, width=90, height=38, command=self.clear2)
        btn_clear_name.grid(row=3, column=1, padx=(10, 20), pady=(0, 20))

        self.lbl_reg_progress = ctk.CTkLabel(reg_card, text="Captured Images: 0 / 100", font=ctk.CTkFont(size=15, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_reg_progress.pack(anchor="w", padx=35, pady=(10, 5))

        self.message1 = ctk.CTkLabel(reg_card, text="1) Take Images  >>>  2) Save Profile", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MUTED)
        self.message1.pack(anchor="w", padx=35, pady=(0, 15))

        reg_actions_bar = ctk.CTkFrame(reg_card, fg_color="transparent")
        reg_actions_bar.pack(anchor="w", padx=30, pady=(0, 20))

        btn_take_img = ctk.CTkButton(
            reg_actions_bar,
            text="Take Images",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLOR_BTN_DARK,
            hover_color=COLOR_BTN_DARK_HOVER,
            text_color=COLOR_WHITE,
            corner_radius=16,
            height=44,
            width=160,
            command=self.TakeImages
        )
        btn_take_img.pack(side="left", padx=(0, 15))

        btn_train_img = ctk.CTkButton(
            reg_actions_bar,
            text="Save Profile (Train)",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLOR_WHITE,
            hover_color="#F3EFEA",
            text_color=COLOR_TEXT_MAIN,
            corner_radius=16,
            height=44,
            width=180,
            command=self.psw
        )
        btn_train_img.pack(side="left")

        # PAGE 3: ATTENDANCE
        page_attend = ctk.CTkFrame(content_container, fg_color="transparent")
        self.pages_dict["Attendance"] = page_attend

        ctk.CTkLabel(page_attend, text="Attendance Tracking", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(page_attend, text="Live facial recognition camera & attendance log records", font=ctk.CTkFont(size=14), text_color=COLOR_TEXT_MUTED).pack(anchor="w", pady=(0, 15))

        attend_control_card = ctk.CTkFrame(page_attend, fg_color=COLOR_CARD_MINT, corner_radius=22)
        attend_control_card.pack(fill="x", pady=(0, 15))

        self.lbl_attend_cam_status = ctk.CTkLabel(attend_control_card, text="Camera: Standby", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_attend_cam_status.pack(side="left", padx=20, pady=15)

        self.lbl_attend_model_status = ctk.CTkLabel(attend_control_card, text="Model Status: Ready", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_MUTED)
        self.lbl_attend_model_status.pack(side="left", padx=15, pady=15)

        btn_track_img = ctk.CTkButton(
            attend_control_card,
            text="Start Attendance Camera",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLOR_BTN_DARK,
            hover_color=COLOR_BTN_DARK_HOVER,
            text_color=COLOR_WHITE,
            corner_radius=16,
            height=42,
            width=210,
            command=self.TrackImages
        )
        btn_track_img.pack(side="right", padx=20, pady=12)

        attend_table_card = ctk.CTkFrame(page_attend, fg_color=COLOR_WHITE, corner_radius=22)
        attend_table_card.pack(fill="both", expand=True)

        attend_tbl_header = ctk.CTkFrame(attend_table_card, fg_color="transparent")
        attend_tbl_header.pack(fill="x", padx=20, pady=(15, 10))

        ctk.CTkLabel(attend_tbl_header, text="Today's Attendance Records", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")
        self.lbl_attend_count = ctk.CTkLabel(attend_tbl_header, text="Total Logs Today: 0", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MUTED)
        self.lbl_attend_count.pack(side="right")

        attend_tbl_container = ctk.CTkFrame(attend_table_card, fg_color="transparent")
        attend_tbl_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.attendance_tree = ttk.Treeview(
            attend_tbl_container,
            columns=('id', 'name', 'date', 'time'),
            show='headings',
            selectmode='browse'
        )
        self.attendance_tree.heading('id', text='STUDENT ID')
        self.attendance_tree.heading('name', text='STUDENT NAME')
        self.attendance_tree.heading('date', text='DATE')
        self.attendance_tree.heading('time', text='LOG TIME')

        self.attendance_tree.column('id', width=130, anchor='center')
        self.attendance_tree.column('name', width=240, anchor='w')
        self.attendance_tree.column('date', width=130, anchor='center')
        self.attendance_tree.column('time', width=130, anchor='center')

        sb_attend = ttk.Scrollbar(attend_tbl_container, orient="vertical", command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=sb_attend.set)

        self.attendance_tree.pack(side="left", fill="both", expand=True)
        sb_attend.pack(side="right", fill="y")

        # PAGE 4: STUDENT RECORDS
        page_records = ctk.CTkFrame(content_container, fg_color="transparent")
        self.pages_dict["Student Records"] = page_records

        ctk.CTkLabel(page_records, text="Student Directory", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(page_records, text="List of all registered students in the system", font=ctk.CTkFont(size=14), text_color=COLOR_TEXT_MUTED).pack(anchor="w", pady=(0, 15))

        records_card = ctk.CTkFrame(page_records, fg_color=COLOR_WHITE, corner_radius=22)
        records_card.pack(fill="both", expand=True)

        rec_header_frame = ctk.CTkFrame(records_card, fg_color="transparent")
        rec_header_frame.pack(fill="x", padx=20, pady=(15, 10))

        self.lbl_records_summary = ctk.CTkLabel(rec_header_frame, text="Total Registered Students: 0", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.lbl_records_summary.pack(side="left")

        btn_refresh_rec = ctk.CTkButton(rec_header_frame, text="Refresh Directory", fg_color=COLOR_SIDEBAR_BG, hover_color="#E2DCD4", text_color=COLOR_TEXT_MAIN, corner_radius=12, height=34, command=self.load_student_records)
        btn_refresh_rec.pack(side="right")

        rec_tbl_container = ctk.CTkFrame(records_card, fg_color="transparent")
        rec_tbl_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.tv_records = ttk.Treeview(rec_tbl_container, columns=('serial', 'student_id', 'student_name'), show='headings', selectmode='browse')
        self.tv_records.heading('serial', text='SERIAL NO.')
        self.tv_records.heading('student_id', text='STUDENT ID')
        self.tv_records.heading('student_name', text='FULL NAME')

        self.tv_records.column('serial', width=100, anchor='center')
        self.tv_records.column('student_id', width=150, anchor='center')
        self.tv_records.column('student_name', width=300, anchor='w')

        sb_records = ttk.Scrollbar(rec_tbl_container, orient="vertical", command=self.tv_records.yview)
        self.tv_records.configure(yscrollcommand=sb_records.set)

        self.tv_records.pack(side="left", fill="both", expand=True)
        sb_records.pack(side="right", fill="y")

        # PAGE 5: SETTINGS
        page_settings = ctk.CTkFrame(content_container, fg_color="transparent")
        self.pages_dict["Settings"] = page_settings

        ctk.CTkLabel(page_settings, text="Settings & Support", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(page_settings, text="Password security, support contacts, and system information", font=ctk.CTkFont(size=14), text_color=COLOR_TEXT_MUTED).pack(anchor="w", pady=(0, 20))

        card_sec = ctk.CTkFrame(page_settings, fg_color=COLOR_CARD_PEACH, corner_radius=22)
        card_sec.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(card_sec, text="Security & Password", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 2))
        ctk.CTkLabel(card_sec, text="Update master admin password for model training authorization.", font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=20, pady=(0, 15))
        ctk.CTkButton(card_sec, text="Change Password", fg_color=COLOR_BTN_DARK, hover_color=COLOR_BTN_DARK_HOVER, text_color=COLOR_WHITE, corner_radius=14, height=38, command=self.change_pass).pack(anchor="w", padx=20, pady=(0, 20))

        card_supp = ctk.CTkFrame(page_settings, fg_color=COLOR_CARD_PINK, corner_radius=22)
        card_supp.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(card_supp, text="Technical Support", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 2))
        ctk.CTkLabel(card_supp, text="Contact the development team for assistance or bug reports.", font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_MUTED).pack(anchor="w", padx=20, pady=(0, 15))
        ctk.CTkButton(card_supp, text="Contact Support", fg_color=COLOR_BTN_DARK, hover_color=COLOR_BTN_DARK_HOVER, text_color=COLOR_WHITE, corner_radius=14, height=38, command=self.contact).pack(anchor="w", padx=20, pady=(0, 20))

        card_about = ctk.CTkFrame(page_settings, fg_color=COLOR_WHITE, corner_radius=22)
        card_about.pack(fill="x")
        ctk.CTkLabel(card_about, text="About Application", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 5))
        ctk.CTkLabel(card_about, text="Face Recognition Attendance System v2.0\nEngine: OpenCV Haar Cascade Classifier & LBPH Face Recognizer\nStorage: Local CSV & YAML Model Files", font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_MUTED, justify="left").pack(anchor="w", padx=20, pady=(0, 15))

    def build_right_panel(self):
        """ Builds quick-info side panel on the right side """
        right_panel_frame = ctk.CTkFrame(self, fg_color=COLOR_PANEL_BG, corner_radius=0)
        right_panel_frame.grid(row=0, column=2, sticky="nsew", padx=0, pady=0)

        time_card = ctk.CTkFrame(right_panel_frame, fg_color=COLOR_WHITE, corner_radius=22)
        time_card.pack(fill="x", padx=20, pady=(25, 15))

        self.date_label = ctk.CTkLabel(time_card, text="", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MUTED)
        self.date_label.pack(anchor="w", padx=20, pady=(15, 0))

        self.clock_label = ctk.CTkLabel(time_card, text="", font=ctk.CTkFont(size=24, weight="bold"), text_color=COLOR_TEXT_MAIN)
        self.clock_label.pack(anchor="w", padx=20, pady=(0, 15))

        info_card = ctk.CTkFrame(right_panel_frame, fg_color=COLOR_WHITE, corner_radius=22)
        info_card.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(info_card, text="System Info", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=20, pady=(12, 6))

        self.lbl_total_regs = ctk.CTkLabel(info_card, text="Total Registrations till now : 0", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MUTED)
        self.lbl_total_regs.pack(anchor="w", padx=20, pady=(0, 12))

    def contact(self):
        """ Displays support contact dialog """
        mess.showinfo(title='Contact Support', message="Face Recognition Attendance System v2.0\nPlease contact support at:\n'anjalitalan19@gmail.com'")

    def tick(self):
        """ Real-time digital clock update loop """
        ts = time.time()
        dt = datetime.datetime.fromtimestamp(ts)
        time_string = dt.strftime('%H:%M:%S')
        day = dt.strftime('%d')
        month_name = dt.strftime('%B')
        year = dt.strftime('%Y')

        date_string = f"{day} {month_name} {year}"

        if hasattr(self, 'clock_label') and self.clock_label.winfo_exists():
            self.clock_label.configure(text=time_string)
        if hasattr(self, 'date_label') and self.date_label.winfo_exists():
            self.date_label.configure(text=date_string)

        self.after(1000, self.tick)

    def refresh_attendance_table(self):
        """ Reloads today's attendance CSV into the attendance and dashboard tables """
        for k in self.attendance_tree.get_children():
            self.attendance_tree.delete(k)
        for k in self.dash_preview_tree.get_children():
            self.dash_preview_tree.delete(k)

        ts = time.time()
        date_str = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        attendance_file = os.path.join("Attendance", f"Attendance_{date_str}.csv")

        record_count = 0
        seen_ids = set()

        if os.path.isfile(attendance_file):
            with open(attendance_file, 'r', encoding='utf-8', errors='ignore') as csvFile1:
                reader1 = csv.reader(csvFile1)
                i = 0
                for lines in reader1:
                    i += 1
                    if i > 1:
                        non_empty = [c.strip() for c in lines if c.strip()]
                        if len(non_empty) >= 4:
                            s_id = normalize_student_id(non_empty[0])
                            s_name = clean_name_str(non_empty[1])
                            s_date = str(non_empty[2]).strip()
                            s_time = str(non_empty[3]).strip()

                            if s_id and s_id not in seen_ids and s_id != "Id":
                                seen_ids.add(s_id)
                                self.attendance_tree.insert('', 'end', values=(s_id, s_name, s_date, s_time))

                                if record_count < 5:
                                    self.dash_preview_tree.insert('', 'end', values=(s_id, s_name, s_date, s_time))

                                record_count += 1

        self.lbl_dash_today.configure(text=f"{record_count} Today")
        self.lbl_attend_count.configure(text=f"Total Logs Today: {record_count}")

    def load_student_records(self):
        """ Reloads deduplicated student records into the Student Directory table """
        for k in self.tv_records.get_children():
            self.tv_records.delete(k)

        records = load_and_clean_student_details()
        unique_student_count = len(records)

        for rec in records:
            self.tv_records.insert('', 'end', values=(rec["serial_no"], rec["student_id"], rec["name"]))

        self.lbl_records_summary.configure(text=f"Total Registered Students: {unique_student_count}")
        self.lbl_dash_students.configure(text=str(unique_student_count))
        self.lbl_total_regs.configure(text=f"Total Registrations till now : {unique_student_count}")

    def update_all_counts(self):
        """ Updates student metrics across dashboard and side panels """
        records = load_and_clean_student_details()
        unique_student_count = len(records)
        self.lbl_dash_students.configure(text=str(unique_student_count))
        self.lbl_total_regs.configure(text=f"Total Registrations till now : {unique_student_count}")
        return unique_student_count

    def update_model_status(self):
        """ Updates status indicator for trained face recognition YML model """
        model_path = os.path.join("TrainingImageLabel", "Trainner.yml")
        if os.path.isfile(model_path):
            self.lbl_dash_model.configure(text="Ready (Trained)")
            self.lbl_attend_model_status.configure(text="Model Status: Ready")
        else:
            self.lbl_dash_model.configure(text="Not Trained")
            self.lbl_attend_model_status.configure(text="Model Status: Not Trained")

    def clear(self):
        """ Clears Student ID input field """
        self.txt_id.delete(0, 'end')
        res = "1) Take Images  >>>  2) Save Profile"
        self.message1.configure(text=res, text_color=COLOR_TEXT_MUTED)

    def clear2(self):
        """ Clears Student Name input field """
        self.txt_name.delete(0, 'end')
        res = "1) Take Images  >>>  2) Save Profile"
        self.message1.configure(text=res, text_color=COLOR_TEXT_MUTED)

    def save_pass_action(self, old_pass, new_pass, confirm_pass, dialog_window):
        """ Validates and updates master admin password """
        assure_path_exists("TrainingImageLabel/")
        password_file = os.path.join("TrainingImageLabel", "psd.txt")

        if os.path.isfile(password_file):
            with open(password_file, "r") as tf:
                key = tf.read().strip()
        else:
            key = ""

        if old_pass == key:
            if new_pass == confirm_pass:
                if new_pass.strip() == "":
                    mess.showwarning(title='Warning', message='Password cannot be empty!')
                    return
                with open(password_file, "w") as txf:
                    txf.write(new_pass)
                mess.showinfo(title='Success', message='Password changed successfully!')
                dialog_window.destroy()
            else:
                mess.showerror(title='Error', message='Confirm new password again!')
        else:
            mess.showerror(title='Wrong Password', message='Please enter correct old password.')

    def change_pass(self):
        """ Opens change password modal dialog """
        assure_path_exists("TrainingImageLabel/")
        password_file = os.path.join("TrainingImageLabel", "psd.txt")

        if not os.path.isfile(password_file):
            dialog = ctk.CTkInputDialog(text="Old password not found.\nEnter new password below:", title="Set Password")
            new_pas = dialog.get_input()
            if new_pas is None or new_pas.strip() == "":
                mess.showwarning(title='Warning', message='Password not set! Please try again.')
            else:
                with open(password_file, "w") as tf:
                    tf.write(new_pas)
                mess.showinfo(title='Success', message='New password registered successfully!')
            return

        pass_window = ctk.CTkToplevel(self)
        pass_window.geometry("420x300")
        pass_window.title("Change Password")
        pass_window.configure(fg_color=COLOR_MAIN_BG)
        pass_window.resizable(False, False)
        pass_window.grab_set()
        pass_window.focus_set()

        lbl_title = ctk.CTkLabel(pass_window, text="Change Password", font=ctk.CTkFont(size=20, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_title.pack(pady=15)

        frame_form = ctk.CTkFrame(pass_window, fg_color=COLOR_WHITE, corner_radius=16)
        frame_form.pack(padx=25, pady=5, fill="x")

        lbl_old = ctk.CTkLabel(frame_form, text="Old Password:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_old.grid(row=0, column=0, sticky="w", pady=8, padx=15)
        entry_old = ctk.CTkEntry(frame_form, show="*", width=200, corner_radius=10, fg_color=COLOR_MAIN_BG, border_color="#D1C7BD", text_color=COLOR_TEXT_MAIN)
        entry_old.grid(row=0, column=1, pady=8, padx=15)

        lbl_new = ctk.CTkLabel(frame_form, text="New Password:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_new.grid(row=1, column=0, sticky="w", pady=8, padx=15)
        entry_new = ctk.CTkEntry(frame_form, show="*", width=200, corner_radius=10, fg_color=COLOR_MAIN_BG, border_color="#D1C7BD", text_color=COLOR_TEXT_MAIN)
        entry_new.grid(row=1, column=1, pady=8, padx=15)

        lbl_confirm = ctk.CTkLabel(frame_form, text="Confirm Password:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_MAIN)
        lbl_confirm.grid(row=2, column=0, sticky="w", pady=8, padx=15)
        entry_confirm = ctk.CTkEntry(frame_form, show="*", width=200, corner_radius=10, fg_color=COLOR_MAIN_BG, border_color="#D1C7BD", text_color=COLOR_TEXT_MAIN)
        entry_confirm.grid(row=2, column=1, pady=8, padx=15)

        btn_frame = ctk.CTkFrame(pass_window, fg_color="transparent")
        btn_frame.pack(pady=20)

        btn_save = ctk.CTkButton(
            btn_frame,
            text="Save",
            fg_color=COLOR_BTN_DARK,
            hover_color=COLOR_BTN_DARK_HOVER,
            text_color=COLOR_WHITE,
            corner_radius=14,
            width=100,
            height=36,
            command=lambda: self.save_pass_action(entry_old.get(), entry_new.get(), entry_confirm.get(), pass_window)
        )
        btn_save.pack(side="left", padx=10)

        btn_cancel = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color=COLOR_SIDEBAR_BG,
            hover_color="#E2DCD4",
            text_color=COLOR_TEXT_MAIN,
            corner_radius=14,
            width=100,
            height=36,
            command=pass_window.destroy
        )
        btn_cancel.pack(side="left", padx=10)

    def psw(self):
        """ Prompts for admin password before training the face model """
        assure_path_exists("TrainingImageLabel/")
        password_file = os.path.join("TrainingImageLabel", "psd.txt")

        if os.path.isfile(password_file):
            with open(password_file, "r") as tf:
                key = tf.read().strip()
        else:
            dialog = ctk.CTkInputDialog(text="Old password not found.\nEnter new password below:", title="Set Password")
            new_pas = dialog.get_input()
            if new_pas is None or new_pas.strip() == "":
                mess.showwarning(title='Warning', message='Password not set! Please try again.')
                return
            else:
                with open(password_file, "w") as tf:
                    tf.write(new_pas)
                mess.showinfo(title='Success', message='New password registered successfully!')
                return

        dialog = ctk.CTkInputDialog(text="Enter Password:", title="Password Required")
        password = dialog.get_input()

        if password == key:
            self.TrainImages()
        elif password is None:
            pass
        else:
            mess.showerror(title='Wrong Password', message='You have entered wrong password')

    def TakeImages(self):
        """ Captures 100 face images using OpenCV webcam feed for student registration """
        if not check_haarcascadefile():
            return

        raw_id = self.txt_id.get()
        Id = normalize_student_id(raw_id)
        name = self.txt_name.get().strip()

        if not Id:
            self.message1.configure(text="Please enter a valid Student ID!", text_color="#DC2626")
            return

        records = load_and_clean_student_details()
        if student_id_exists(Id, records):
            err_msg = f"Student ID {Id} is already registered.\nPlease enter a unique Student ID."
            self.message1.configure(text=f"Student ID {Id} is already registered!", text_color="#DC2626")
            mess.showwarning(title="Duplicate Student ID", message=err_msg)
            return

        if not (name.isalpha() or (' ' in name and name.replace(' ', '').isalpha())):
            self.message1.configure(text="Enter Correct name (letters & spaces only)", text_color="#DC2626")
            return

        serial = get_next_serial_number(records)

        columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
        assure_path_exists("StudentDetails/")
        assure_path_exists("TrainingImage/")

        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        self.lbl_dash_camera.configure(text="Active")
        self.lbl_attend_cam_status.configure(text="Camera: Capturing")

        while True:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                img_name = f" {name}.{serial}.{Id}.{sampleNum}.jpg"
                save_path = os.path.join("TrainingImage", img_name)
                cv2.imwrite(save_path, gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images - Press Q to Cancel', img)

                self.lbl_reg_progress.configure(text=f"Captured Images: {sampleNum} / 100")
                self.update_idletasks()

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            elif sampleNum >= 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        self.lbl_dash_camera.configure(text="Standby")
        self.lbl_attend_cam_status.configure(text="Camera: Standby")

        if sampleNum > 0:
            res = f"Images Taken for ID : {Id}"
            row = [serial, '', Id, '', name]
            csv_path = os.path.join("StudentDetails", "StudentDetails.csv")
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, 'a+', newline='', encoding='utf-8') as csvFile:
                writer = csv.writer(csvFile)
                if not file_exists:
                    writer.writerow(columns)
                writer.writerow(row)

            self.message1.configure(text=res, text_color="#059669")
            self.lbl_reg_progress.configure(text=f"Captured Images: {sampleNum} / 100 (Completed)")
            self.update_all_counts()
            self.load_student_records()
        else:
            self.message1.configure(text="No images captured.", text_color="#DC2626")

    def TrainImages(self):
        """ Trains LBPH face recognizer model on captured images """
        if not check_haarcascadefile():
            return
        assure_path_exists("TrainingImageLabel/")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ID = getImagesAndLabels("TrainingImage")

        if len(faces) == 0:
            mess.showwarning(title='No Registrations', message='Please Register someone first!!!')
            return

        try:
            recognizer.train(faces, np.array(ID))
        except Exception:
            mess.showwarning(title='No Registrations', message='Please Register someone first!!!')
            return

        model_path = os.path.join("TrainingImageLabel", "Trainner.yml")
        recognizer.save(model_path)
        res = "Profile Saved Successfully"
        self.message1.configure(text=res, text_color="#059669")
        self.update_all_counts()
        self.update_model_status()
        mess.showinfo(title='Success', message='Face recognition model trained successfully!')

    def get_existing_today_ids(self, attendance_path):
        """ Reads today's logged Student IDs to prevent duplicate attendance logs """
        existing_ids = set()
        if os.path.isfile(attendance_path):
            with open(attendance_path, 'r', encoding='utf-8', errors='ignore') as csvFile1:
                reader1 = csv.reader(csvFile1)
                i = 0
                for lines in reader1:
                    i += 1
                    if i > 1:
                        non_empty = [c.strip() for c in lines if c.strip()]
                        if len(non_empty) >= 1:
                            s_id = normalize_student_id(non_empty[0])
                            if s_id and s_id != "Id":
                                existing_ids.add(s_id)
        return existing_ids

    def TrackImages(self):
        """ Launches real-time video feed for facial recognition and logs attendance """
        if not check_haarcascadefile():
            return
        assure_path_exists("Attendance/")
        assure_path_exists("StudentDetails/")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        model_path = os.path.join("TrainingImageLabel", "Trainner.yml")
        if os.path.isfile(model_path):
            recognizer.read(model_path)
        else:
            mess.showwarning(title='Data Missing', message='Please click on Save Profile to reset data!!')
            return

        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)

        student_records = load_and_clean_student_details()
        if not student_records:
            mess.showerror(title='Details Missing', message='Student details are missing, please check!')
            return

        ts = time.time()
        date_str = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        attendance_path = os.path.join("Attendance", f"Attendance_{date_str}.csv")

        existing_today_ids = self.get_existing_today_ids(attendance_path)
        session_recognized_ids = set()

        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']

        self.lbl_dash_camera.configure(text="Active")
        self.lbl_attend_cam_status.configure(text="Camera: Tracking Active")

        new_attendance_logged = False
        already_marked_names = set()

        while True:
            ret, im = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

                if conf < 50:
                    ts_now = time.time()
                    timeStamp = datetime.datetime.fromtimestamp(ts_now).strftime('%H:%M:%S')

                    st = get_student_by_serial(serial, student_records)
                    if st:
                        id_val = st["student_id"]
                        name_val = st["name"]
                    else:
                        id_val = "Unknown"
                        name_val = "Unknown"

                    bb = name_val

                    if id_val != "Unknown":
                        if id_val not in session_recognized_ids and id_val not in existing_today_ids:
                            session_recognized_ids.add(id_val)
                            existing_today_ids.add(id_val)

                            record = [id_val, '', name_val, '', str(date_str), '', str(timeStamp)]
                            file_exists = os.path.isfile(attendance_path)
                            with open(attendance_path, 'a+', newline='', encoding='utf-8') as csvFile1:
                                writer = csv.writer(csvFile1)
                                if not file_exists:
                                    writer.writerow(col_names)
                                writer.writerow(record)

                            new_attendance_logged = True
                            self.after(0, self.refresh_attendance_table)

                        elif id_val in existing_today_ids and name_val not in already_marked_names:
                            already_marked_names.add(name_val)
                else:
                    bb = 'Unknown'

                cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)

            cv2.imshow('Taking Attendance - Press Q to Stop', im)
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        self.lbl_dash_camera.configure(text="Standby")
        self.lbl_attend_cam_status.configure(text="Camera: Standby")

        self.after(0, self.refresh_attendance_table)

        if already_marked_names and not new_attendance_logged:
            marked_list = ", ".join(already_marked_names)
            mess.showinfo(title='Attendance Info', message=f'Attendance already marked for {marked_list} today.')


if __name__ == '__main__':
    app = AttendanceApp()
    app.mainloop()
