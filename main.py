import os
from pathlib import Path
import sys
import time
import math
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QProgressBar, QMessageBox
)

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MODEL_PATH = str((Path(__file__).resolve().parent / "models" / "hand_landmarker.task"))

DETECTION_CONFIDENCE = 0.60
PRESENCE_CONFIDENCE = 0.60
TRACKING_CONFIDENCE = 0.60
CURSOR_SMOOTHING = 0.22
PINCH_THRESHOLD = 0.055
CLICK_COOLDOWN = 0.70
ACTION_COOLDOWN = 1.00
SWIPE_DISTANCE = 0.25
SWIPE_COOLDOWN = 1.20

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

class VisionControlCenter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISION CONTROL CENTER")
        self.resize(1450, 900)
        self.running = True
        self.control_enabled = True
        self.camera = None
        self.landmarker = None
        self.last_timestamp = 0
        self.dragging = False
        self.cursor_x = SCREEN_WIDTH // 2
        self.cursor_y = SCREEN_HEIGHT // 2
        self.last_click = 0
        self.last_action = 0
        self.last_swipe = 0
        self.last_brightness_update = 0
        self.last_brightness_level = None
        self.brightness_unavailable = False
        self.history = deque(maxlen=12)
        self.total_commands = 0
        self.capture_dir = (Path(__file__).resolve().parent / "captures")
        self.photo_dir = self.capture_dir / "photos"
        self.screenshot_dir = self.capture_dir / "screenshots"
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.last_frame = None
        self.current_gesture = "WAITING"
        self.current_action = "Waiting for hand"
        self.confidence = 0.0
        self.frame_counter = 0
        self.fps = 0
        self.fps_start = time.time()
        self.build_ui()
        self.initialize_camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(15)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("""
            QWidget { background:#080c11; color:#edf2f7; font-family:"Segoe UI"; }
            QFrame#card { background:#111820; border:1px solid #26333f; border-radius:16px; }
            QLabel#title { font-size:26px; font-weight:700; color:#fff; }
            QLabel#subtitle { font-size:13px; color:#718092; }
            QLabel#section { font-size:12px; font-weight:700; color:#718092; }
            QLabel#value { font-size:22px; font-weight:700; color:#25d6a0; }
            QLabel#small { font-size:12px; color:#91a0ae; }
            QPushButton { background:#17212b; color:#e9eef3; border:1px solid #30404e;
                          border-radius:9px; padding:11px; font-weight:600; }
            QPushButton:hover { background:#202d39; border:1px solid #25d6a0; }
            QPushButton#active { background:#123b30; border:1px solid #25d6a0; }
            QProgressBar { background:#1b242d; border:none; border-radius:5px; height:8px; }
            QProgressBar::chunk { background:#25d6a0; border-radius:5px; }
        """)
        main = QVBoxLayout(central)
        main.setContentsMargins(28,24,28,20)
        main.setSpacing(18)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("VISION CONTROL CENTER"); title.setObjectName("title")
        subtitle = QLabel("AI-powered touchless computer interaction"); subtitle.setObjectName("subtitle")
        title_box.addWidget(title); title_box.addWidget(subtitle)
        header.addLayout(title_box); header.addStretch()
        self.system_status = QLabel("● SYSTEM ACTIVE")
        self.system_status.setStyleSheet("color:#25d6a0;font-size:13px;font-weight:700;")
        header.addWidget(self.system_status)
        main.addLayout(header)

        content = QHBoxLayout(); content.setSpacing(18)

        camera_card = QFrame(); camera_card.setObjectName("card")
        camera_layout = QVBoxLayout(camera_card)
        camera_header = QHBoxLayout()
        camera_title = QLabel("LIVE VISION FEED"); camera_title.setObjectName("section")
        self.fps_label = QLabel("FPS --"); self.fps_label.setObjectName("small")
        camera_header.addWidget(camera_title); camera_header.addStretch(); camera_header.addWidget(self.fps_label)
        camera_layout.addLayout(camera_header)
        self.camera_view = QLabel(); self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(850,600)
        self.camera_view.setStyleSheet("background:#05080b;border:1px solid #26333f;border-radius:12px;")
        camera_layout.addWidget(self.camera_view)
        content.addWidget(camera_card, 3)

        side = QVBoxLayout(); side.setSpacing(14)

        gesture_card = QFrame(); gesture_card.setObjectName("card")
        gl = QVBoxLayout(gesture_card)
        section = QLabel("CURRENT GESTURE"); section.setObjectName("section")
        self.gesture_value = QLabel("WAITING"); self.gesture_value.setObjectName("value")
        self.action_value = QLabel("Waiting for hand"); self.action_value.setObjectName("small")
        confidence = QLabel("CONFIDENCE"); confidence.setObjectName("section")
        self.confidence_bar = QProgressBar(); self.confidence_bar.setRange(0,100); self.confidence_bar.setValue(0)
        gl.addWidget(section); gl.addSpacing(5); gl.addWidget(self.gesture_value); gl.addWidget(self.action_value)
        gl.addSpacing(15); gl.addWidget(confidence); gl.addWidget(self.confidence_bar)
        side.addWidget(gesture_card)

        analytics_card = QFrame(); analytics_card.setObjectName("card")
        al = QVBoxLayout(analytics_card)
        at = QLabel("SESSION ANALYTICS"); at.setObjectName("section")
        self.commands_value = QLabel("0"); self.commands_value.setObjectName("value")
        ct = QLabel("COMMANDS EXECUTED"); ct.setObjectName("small")
        self.tracking_value = QLabel("SEARCHING"); self.tracking_value.setObjectName("value")
        tt = QLabel("HAND TRACKING"); tt.setObjectName("small")
        al.addWidget(at); al.addSpacing(8); al.addWidget(self.commands_value); al.addWidget(ct)
        al.addSpacing(12); al.addWidget(self.tracking_value); al.addWidget(tt)
        side.addWidget(analytics_card)

        control_card = QFrame(); control_card.setObjectName("card")
        cl = QVBoxLayout(control_card)
        ctl = QLabel("ACTIVITY CENTER"); ctl.setObjectName("section")
        self.pause_button = QPushButton("PAUSE CONTROL"); self.pause_button.setObjectName("active")
        self.pause_button.clicked.connect(self.toggle_control)
        self.photo_button = QPushButton("SAVE PHOTO")
        self.photo_button.clicked.connect(self.capture_photo_now)
        self.screenshot_button = QPushButton("SAVE SCREENSHOT")
        self.screenshot_button.clicked.connect(self.capture_screenshot_now)
        self.activity_button = QPushButton("OPEN ACTIVITY CENTER")
        self.activity_button.clicked.connect(self.open_activity_center)
        exit_button = QPushButton("EXIT SYSTEM"); exit_button.clicked.connect(self.close)
        cl.addWidget(ctl); cl.addSpacing(5); cl.addWidget(self.pause_button)
        cl.addWidget(self.photo_button); cl.addWidget(self.screenshot_button)
        cl.addWidget(self.activity_button); cl.addWidget(exit_button)
        side.addWidget(control_card)

        map_card = QFrame(); map_card.setObjectName("card")
        ml = QVBoxLayout(map_card)
        mt = QLabel("GESTURE COMMANDS"); mt.setObjectName("section"); ml.addWidget(mt)
        for icon, text in [
            ("☝", "Cursor movement"), ("👍", "Volume up"), ("👎", "Volume down"),
            ("🖐", "Brightness control"), ("🤏", "Screenshot"), ("👌", "Photo")
        ]:
            row = QHBoxLayout()
            a = QLabel(icon); a.setStyleSheet("font-size:16px;")
            b = QLabel(text); b.setObjectName("small")
            row.addWidget(a); row.addWidget(b); row.addStretch(); ml.addLayout(row)
        side.addWidget(map_card)
        content.addLayout(side,1)
        main.addLayout(content)

        footer = QHBoxLayout()
        ft = QLabel("OpenCV  •  MediaPipe Tasks  •  PyAutoGUI"); ft.setObjectName("small")
        footer.addWidget(ft); footer.addStretch()
        esc = QLabel("ESC  Emergency Exit"); esc.setObjectName("small"); footer.addWidget(esc)
        main.addLayout(footer)

    def initialize_camera(self):
        self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not self.camera.isOpened():
            QMessageBox.critical(self,"Camera Error","Could not open the webcam.")
            self.close(); return
        if not os.path.exists(MODEL_PATH):
            QMessageBox.critical(self,"Model Missing",
                f"MediaPipe model not found:\n\n{MODEL_PATH}\n\n"
                "Place hand_landmarker.task in the models folder.")
            self.close(); return
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=PRESENCE_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        try:
            self.landmarker = HandLandmarker.create_from_options(options)
        except Exception as error:
            QMessageBox.critical(self,"MediaPipe Error",str(error))
            self.close()

    def finger_states(self,lm):
        index = lm[8].y < lm[6].y and lm[8].y < lm[5].y
        middle = lm[12].y < lm[10].y and lm[12].y < lm[9].y
        ring = lm[16].y < lm[14].y and lm[16].y < lm[13].y
        pinky = lm[20].y < lm[18].y and lm[20].y < lm[17].y
        thumb = distance(lm[4],lm[0]) > distance(lm[3],lm[0])
        return thumb,index,middle,ring,pinky

    def classify_gesture(self,lm):
        thumb,index,middle,ring,pinky = self.finger_states(lm)
        pinch = distance(lm[4],lm[8])
        palm_open = index and middle and ring and pinky
        all_fingers_closed = not index and not middle and not ring and not pinky

        if pinch < PINCH_THRESHOLD:
            if middle and ring and pinky: return "OK Sign",0.94
            return "Pinch",0.95
        if palm_open and abs(lm[8].x - lm[4].x) < 0.38 and abs(lm[8].y - lm[12].y) < 0.38:
            return "Open Palm",0.96
        if index and middle and not ring and not pinky:
            return "Two Finger",0.93
        if index and not middle and not ring and not pinky:
            return "Index",0.94
        if all_fingers_closed and thumb:
            return ("Thumb Up",0.92) if lm[4].y < lm[3].y else ("Thumb Down",0.90)
        if all_fingers_closed: return "Fist",0.91
        return "Unknown",0.50

    def move_cursor(self,lm):
        x,y=lm[8].x,lm[8].y
        sx=np.interp(x,[0.10,0.90],[0,SCREEN_WIDTH])
        sy=np.interp(y,[0.10,0.90],[0,SCREEN_HEIGHT])
        sx=clamp(sx,0,SCREEN_WIDTH-1); sy=clamp(sy,0,SCREEN_HEIGHT-1)
        self.cursor_x += (sx-self.cursor_x)*CURSOR_SMOOTHING
        self.cursor_y += (sy-self.cursor_y)*CURSOR_SMOOTHING
        pyautogui.moveTo(int(self.cursor_x),int(self.cursor_y),duration=0)

    def open_activity_center(self):
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(self.capture_dir))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(self.capture_dir)])
            self.current_action = "Activity Center opened"
        except Exception:
            self.current_action = "Could not open Activity Center"

    def capture_photo_now(self):
        frame = self.last_frame.copy() if self.last_frame is not None else None
        if frame is None:
            self.current_action = "No photo frame available"
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.photo_dir / f"photo_{timestamp}.png"
        cv2.imwrite(str(path), frame)
        self.current_action = f"Photo saved: {path.name}"
        self.register_command("Photo")

    def capture_screenshot_now(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.screenshot_dir / f"screenshot_{timestamp}.png"
        pyautogui.screenshot().save(path)
        self.current_action = f"Screenshot saved: {path.name}"
        self.register_command("Screenshot")

    def save_photo(self, frame):
        if frame is None:
            frame = self.last_frame.copy() if self.last_frame is not None else None
        if frame is None:
            self.current_action = "No photo frame available"
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.photo_dir / f"photo_{timestamp}.png"
        cv2.imwrite(str(path), frame)
        self.current_action = f"Photo saved: {path.name}"
        self.register_command("Photo")

    def save_screenshot(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.screenshot_dir / f"screenshot_{timestamp}.png"
        pyautogui.screenshot().save(path)
        self.current_action = f"Screenshot saved: {path.name}"
        self.register_command("Screenshot")

    def execute_command(self,gesture):
        now=time.time()
        if gesture=="Pinch" and now-self.last_action>ACTION_COOLDOWN:
            self.capture_screenshot_now(); self.last_action=now
        elif gesture=="OK Sign" and now-self.last_action>ACTION_COOLDOWN:
            self.capture_photo_now(); self.last_action=now
        elif gesture=="Thumb Up" and now-self.last_action>ACTION_COOLDOWN:
            pyautogui.press("volumeup"); self.last_action=now; self.register_command("Volume Up")
        elif gesture=="Thumb Down" and now-self.last_action>ACTION_COOLDOWN:
            pyautogui.press("volumedown"); self.last_action=now; self.register_command("Volume Down")

    def control_brightness(self,lm):
        if sbc is None or self.brightness_unavailable:
            self.current_action="Brightness control unavailable"
            return
        level=int(clamp(np.interp(lm[9].y,[0.15,0.85],[100,0]),0,100))
        now=time.time()
        if (self.last_brightness_level is not None
                and abs(level-self.last_brightness_level)<3):
            return
        if now-self.last_brightness_update<0.15:
            return
        try:
            sbc.set_brightness(level)
        except Exception:
            self.brightness_unavailable=True
            self.current_action="Brightness control unavailable"
            return
        self.last_brightness_level=level
        self.last_brightness_update=now
        self.current_action=f"Brightness {level}%"
        self.register_command("Brightness")

    def handle_drag(self,gesture):
        if gesture=="Fist":
            if not self.dragging:
                pyautogui.mouseDown(); self.dragging=True; self.register_command("Drag")
        elif self.dragging:
            pyautogui.mouseUp(); self.dragging=False

    def detect_swipe(self):
        if len(self.history)<10 or time.time()-self.last_swipe<SWIPE_COOLDOWN: return
        movement=self.history[-1]-self.history[0]
        if movement>SWIPE_DISTANCE:
            pyautogui.press("right"); self.last_swipe=time.time(); self.register_command("Swipe")
            self.current_action="NEXT"; self.history.clear()
        elif movement<-SWIPE_DISTANCE:
            pyautogui.press("left"); self.last_swipe=time.time(); self.register_command("Swipe")
            self.current_action="PREVIOUS"; self.history.clear()

    def register_command(self,command):
        self.total_commands += 1
        self.current_action=command

    def process_frame(self):
        if not self.running: return
        ok,frame=self.camera.read()
        if not ok: return
        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        timestamp=int(time.time()*1000)
        if timestamp<=self.last_timestamp: timestamp=self.last_timestamp+1
        self.last_timestamp=timestamp
        try:
            result=self.landmarker.detect_for_video(mp_image,timestamp)
        except Exception:
            return
        hand_found=False
        if result.hand_landmarks:
            hand_found=True
            lm=result.hand_landmarks[0]
            gesture,confidence=self.classify_gesture(lm)
            self.current_gesture=gesture; self.confidence=confidence
            self.history.append(lm[8].x)
            self.last_frame = frame.copy()
            self.draw_hand(frame,lm)
            if self.control_enabled:
                if gesture=="Index":
                    self.move_cursor(lm); self.current_action="Cursor Control"
                elif gesture=="Open Palm":
                    self.control_brightness(lm)
                elif gesture in ["Thumb Up", "Thumb Down", "Pinch", "OK Sign"]:
                    self.execute_command(gesture)
                elif gesture in ["Fist", "Unknown"]:
                    self.current_action="Waiting for active gesture"
            else:
                self.current_action="Control Paused"
        else:
            self.current_gesture="NO HAND"; self.confidence=0; self.current_action="Waiting for hand"
            self.history.clear()
            if self.dragging: pyautogui.mouseUp(); self.dragging=False
        self.draw_overlay(frame,hand_found)
        self.frame_counter+=1
        elapsed=time.time()-self.fps_start
        if elapsed>=1:
            self.fps=self.frame_counter/elapsed; self.frame_counter=0; self.fps_start=time.time()
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb_frame.shape
        image=QImage(rgb_frame.data,w,h,ch*w,QImage.Format_RGB888)
        pixmap=QPixmap.fromImage(image).scaled(self.camera_view.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.camera_view.setPixmap(pixmap)
        self.update_dashboard()

    def draw_hand(self,frame,lm):
        h,w=frame.shape[:2]; pts=[]
        for p in lm:
            x=int(p.x*w); y=int(p.y*h); pts.append((x,y))
            cv2.circle(frame,(x,y),4,(80,220,170),-1)
        for a,b in HAND_CONNECTIONS:
            cv2.line(frame,pts[a],pts[b],(70,150,120),2,cv2.LINE_AA)

    def draw_overlay(self,frame,found):
        h,w=frame.shape[:2]
        status="CONTROL ACTIVE" if self.control_enabled else "CONTROL PAUSED"
        cv2.rectangle(frame,(20,20),(330,70),(8,13,18),-1)
        cv2.putText(frame,status,(38,52),cv2.FONT_HERSHEY_SIMPLEX,.72,(80,220,170),2,cv2.LINE_AA)
        cv2.putText(frame,"TRACKING" if found else "SEARCHING",(w-180,42),cv2.FONT_HERSHEY_SIMPLEX,.55,(180,195,205),2,cv2.LINE_AA)
        cv2.putText(frame,f"Gesture: {self.current_gesture}",(25,h-70),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,240,245),2,cv2.LINE_AA)
        cv2.putText(frame,f"Confidence: {int(self.confidence*100)}%",(25,h-35),cv2.FONT_HERSHEY_SIMPLEX,.55,(155,170,185),2,cv2.LINE_AA)

    def update_dashboard(self):
        self.fps_label.setText(f"FPS {self.fps:.0f}")
        self.gesture_value.setText(self.current_gesture)
        self.action_value.setText(self.current_action)
        self.confidence_bar.setValue(int(self.confidence*100))
        self.commands_value.setText(str(self.total_commands))
        self.tracking_value.setText("SEARCHING" if self.current_gesture=="NO HAND" else "ACTIVE")

    def toggle_control(self):
        self.control_enabled=not self.control_enabled
        if self.control_enabled:
            self.pause_button.setText("PAUSE CONTROL")
            self.system_status.setText("● SYSTEM ACTIVE")
        else:
            if self.dragging: pyautogui.mouseUp(); self.dragging=False
            self.pause_button.setText("RESUME CONTROL")
            self.system_status.setText("● CONTROL PAUSED")

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Escape: self.close()

    def closeEvent(self,event):
        self.running=False
        if self.dragging: pyautogui.mouseUp()
        if self.timer: self.timer.stop()
        if self.camera: self.camera.release()
        if self.landmarker: self.landmarker.close()
        event.accept()

def main():
    pyautogui.FAILSAFE=True
    pyautogui.PAUSE=.03
    app=QApplication(sys.argv)
    app.setApplicationName("Vision Control Center")
    window=VisionControlCenter()
    window.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()

#--- py -3.13 -m venv .venv

#--- .\.venv\Scripts\Activate.ps1

#--- python -m pip install -r requirements.txt

#--- python main.py