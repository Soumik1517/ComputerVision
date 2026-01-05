import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import json, time, os
from itertools import combinations
import sys

def get_app_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

fdb = os.path.join(get_app_path(), "face_db.json")
db = "face_db.json"
if os.path.exists(fdb)!=True:
    f = open(fdb,"w")
    json.dump({}, f, indent=2)
    f.close()

# ---------- mediapipe ----------
mpf = mp.solutions.face_mesh
mpd = mp.solutions.drawing_utils
fm = mpf.FaceMesh(False, 1, True, 0.5, 0.5)
pts = [10,33,133,362,263,1,13,61,291,199,152] #landmarks
pairs = list(combinations(pts,2))
cap = None

# -----aka jhoka-------
def draw_minimal(frame, lm):
    h = frame.shape[0]
    w = frame.shape[1]
    for i in pts:
        x = int(lm[i].x * w)
        y = int(lm[i].y * h)
        cv2.circle(frame, (x, y), 2, (0,255,0), -1)

# ---------- register face ----------
def register_face(name, uid, info):
    global cap
    f = open(fdb,"r")
    try:
        db = json.load(f)
    except:
        db = {}
    f.close()

    if name in db:
        messagebox.showinfo("Register", name + " exists")
        return

    cv2.namedWindow("Enroll")
    while True:
        r, fr = cap.read()
        if not r:
            break
        fr = cv2.flip(fr, 1)

        # countdown 
        for i in range(2,0,-1):
            t0 = time.time()
            while time.time() - t0 < 1:
                cv2.waitKey(10)
                ok, live = cap.read()
                if not ok:
                    continue
                live = cv2.flip(live,1)
                cv2.putText(live,"Capturing in "+str(i),(30,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
                cv2.imshow("Enroll", live)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # capturing 
        ok, final = cap.read()
        if not ok:
            continue
        final = cv2.flip(final,1)
        res = fm.process(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            messagebox.showinfo("Register", "No face detected")
            return
        lm = res.multi_face_landmarks[0].landmark
        draw_minimal(final, lm)
        cv2.imshow("Enroll", final)
        cv2.waitKey(2000)
        break

    h = final.shape[0]
    w = final.shape[1]
    p = []
    for l in lm:
        x = int(l.x * w)
        y = int(l.y * h)
        p.append((x,y))

    sel = []
    for i in pts:
        sel.append(p[i])
    xs = []
    for x,y in sel:
        xs.append(x)
    fw = max(xs) - min(xs)
    if fw == 0:
        fw = 1

    emb = []
    for i,j in pairs:
        dx = p[i][0] - p[j][0]
        dy = p[i][1] - p[j][1]
        d = (dx*dx + dy*dy) ** 0.5
        emb.append(d/fw)

    db[name] = {"id":uid, "info":info, "emb":emb, "t":time.time()}
    f = open(fdb,"w")
    json.dump(db,f,indent=2)
    f.close()
    cv2.destroyWindow("Enroll")
    messagebox.showinfo("Register", name + " enrolled")

# ---------- verification ----------
def verify_face():
    global cap
    f = open(fdb,"r")
    try:
        db = json.load(f)
    except:
        db = {}
    f.close()
    if not db:
        messagebox.showinfo("Verify","Empty DB")
        return

    cv2.namedWindow("Verify")
    while True:
        r, fr = cap.read()
        if not r:
            break
        fr = cv2.flip(fr,1)

        # countdown
        for i in range(2,0,-1):
            t0 = time.time()
            while time.time() - t0 < 1:
                cv2.waitKey(10)
                ok, live = cap.read()
                if not ok:
                    continue
                live = cv2.flip(live,1)
                cv2.putText(live,"Capturing in "+str(i),(30,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
                cv2.imshow("Verify", live)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # capturing the frame
        ok, final = cap.read()
        if not ok:
            continue
        final = cv2.flip(final,1)
        res = fm.process(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            messagebox.showinfo("Verify","No face detected")
            return
        lm = res.multi_face_landmarks[0].landmark
        draw_minimal(final, lm)
        cv2.imshow("Verify", final)
        cv2.waitKey(2000)
        break

    h = final.shape[0]
    w = final.shape[1]
    p = []
    for l in lm:
        x = int(l.x * w)
        y = int(l.y * h)
        p.append((x,y))

    sel = []
    for i in pts:
        sel.append(p[i])
    xs = []
    for x,y in sel:
        xs.append(x)
    fw = max(xs) - min(xs)
    if fw == 0:
        fw = 1

    emb = []
    for i,j in pairs:
        dx = p[i][0] - p[j][0]
        dy = p[i][1] - p[j][1]
        d = (dx*dx + dy*dy) ** 0.5
        emb.append(d/fw)

    best = None
    bs = 1000
    for u,d in db.items():
        s = 0
        for a,b in zip(emb,d["emb"]):
            s += abs(a-b)
        s = s / len(emb)
        if s < bs:
            bs = s
            best = u

    rf = final.copy()
    th = 0.05
    if bs < th:
        info = db[best]
        cv2.putText(rf,"Verified:"+best,(30,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(rf,"ID:"+info['id'],(30,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(rf,"Info:"+info['info'],(30,120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.imshow("Verify",rf)
        cv2.waitKey(2000)
        cv2.destroyWindow("Verify")
        messagebox.showinfo("Verify", best+"\nID:"+info['id']+"\nInfo:"+info['info'])
    else:
        cv2.putText(rf,"Not found",(30,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.imshow("Verify",rf)
        cv2.waitKey(2000)

# ---------- GUI ----------
def start_registration():
    root.withdraw()
    global cap
    cap = cv2.VideoCapture(0)
    if not cap:
        messagebox.showinfo("Error","Camera failed")
        root.deiconify()
        return

    rw = tk.Toplevel(root)
    rw.title("Register")
    rw.geometry("300x200")

    tk.Label(rw,text="Name").grid(row=0,column=0,padx=5,pady=5)
    n = tk.Entry(rw)
    n.grid(row=0,column=1,padx=5,pady=5)

    tk.Label(rw,text="ID").grid(row=1,column=0,padx=5,pady=5)
    i = tk.Entry(rw)
    i.grid(row=1,column=1,padx=5,pady=5)

    tk.Label(rw,text="Info").grid(row=2,column=0,padx=5,pady=5)
    inf = tk.Entry(rw)
    inf.grid(row=2,column=1,padx=5,pady=5)

    def ok():
        nm = n.get()
        uid = i.get()
        info = inf.get()
        if not nm:
            messagebox.showinfo("Error","Enter name")
        else:
            rw.destroy()
            register_face(nm, uid, info)
            cap.release()
            cv2.destroyAllWindows()
            root.deiconify()

    def cancel():
        rw.destroy()
        cap.release()
        cv2.destroyAllWindows()
        root.deiconify()

    tk.Button(rw,text="Start",command=ok).grid(row=3,column=0,padx=5,pady=10)
    tk.Button(rw,text="Cancel",command=cancel).grid(row=3,column=1,padx=5,pady=10)

def start_verification():
    root.withdraw()
    global cap
    cap = cv2.VideoCapture(0)
    if not cap:
        messagebox.showinfo("Error","Camera failed")
        root.deiconify()
        return

    verify_face()
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()

root = tk.Tk()
root.title("Face Verification")
root.geometry("320x220")

tk.Label(root,text="Face Verification",font=("Arial",14)).grid(row=0,column=0,columnspan=2,pady=10)
tk.Button(root,text="Register",width=20,command=start_registration).grid(row=1,column=0,columnspan=2,pady=5)
tk.Button(root,text="Verify",width=20,command=start_verification).grid(row=2,column=0,columnspan=2,pady=5)
tk.Button(root,text="Quit",width=20,command=root.quit).grid(row=3,column=0,columnspan=2,pady=5)
root.mainloop()
