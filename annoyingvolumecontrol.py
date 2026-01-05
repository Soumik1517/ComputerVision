import cv2
import os
import mediapipe as mp
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
import comtypes
from comtypes import CLSCTX_ALL
# ------------- SETUP ------------- #
cap = cv2.VideoCapture(0)
prev=None
volume=0
openwidth=None
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
starttime=time.time()
face = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
volumecontrol = cast(interface, POINTER(IAudioEndpointVolume))

def setvolume(smooth_pct):
    smooth_pct = max(0, min(100, smooth_pct))
    
    if smooth_pct <5:
        volumecontrol.SetMasterVolumeLevelScalar(0.01,None)
    else:
        volumecontrol.SetMasterVolumeLevelScalar(smooth_pct / 100.0, None)

def calculation(frame):

    
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    out = face.process(rgb)

    if out.multi_face_landmarks:
        lm = out.multi_face_landmarks[0].landmark

        
        upper = lm[13].y * h       # upper lip cen
        lower = lm[14].y * h      # lower lip cen

        
        left = lm[61].x * w     # left lip cor
        right = lm[291].x * w   # right lip cor

        fw = abs((lm[454].x * w) - (lm[234].x * w))
        pixelgap = abs(upper - lower)
        

        norm_gap = pixelgap/fw
        return norm_gap
    


        print()


    if cv2.waitKey(1) & 0xFF == 27:
        return


while True:
    sucess, frame = cap.read()
    if sucess==False:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Volume Control", frame)

    value = calculation(frame)
    if value == None:
        print("no face detected")
        continue

    currenttime = time.time()
    openwidth
    if openwidth is None:
        if currenttime - starttime > 2:
            openwidth = value
            print("Calibrated open mouth:", openwidth)
        else:
            print("waiting for calibration...")
            continue

    # compute percentage
    pct = (value / openwidth) * 100
    if prev is None:
        smoothed_pct = pct
    else:
        smoothed_pct = prev * 0.7 + pct * 0.3
    prev = smoothed_pct
    smoothed_pct = max(0, min(150, smoothed_pct))
    smoothed_pct = round(smoothed_pct / 7) * 7
    print(int(smoothed_pct))
    if smoothed_pct !=volume:
        setvolume(smoothed_pct)
        volume=smoothed_pct

cap.release()
cv2.destroyAllWindows()
