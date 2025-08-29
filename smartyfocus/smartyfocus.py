import cv2
import mediapipe as mp
import tkinter as tk
import threading
import time
import matplotlib.pyplot as plt
import pygetwindow as gw 


focused_time = 0
distracted_time = 0
timer_running = False
exit_app = False
target_app = "" 

def is_face_detected_mediapipe(frame):
    """
    Uses MediaPipe to detect a face in the given frame.
    Returns True if a face is found, False otherwise.
    """
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections:
            return True
        return False

def get_active_window_title():
    """
    Uses pygetwindow to get the title of the currently active window.
    Returns an empty string if no window is active.
    """
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            return active_window.title
        else:
            return ""
    except gw.PyGetWindowException:
        
        return ""

def webcam_loop():
    """
    Main loop to capture video, detect face, and track focus time.
    It now also checks if the target application is in the foreground.
    """
    global focused_time, distracted_time, timer_running, exit_app
    cap = cv2.VideoCapture(0)
    last_check = time.time()

    
    while not target_app and not exit_app:
        time.sleep(0.1)

    while not exit_app:
        ret, frame = cap.read()
        if not ret:
            break

        
        face_detected = is_face_detected_mediapipe(frame)

        
        current_active_window = get_active_window_title()
        app_is_active = (current_active_window == target_app)

        now = time.time()
        
        
        if face_detected and app_is_active:
            timer_running = True
            focused_time += now - last_check
            status = "Focused on " + target_app
            color = (0, 255, 0) 
        else:
            timer_running = False
            distracted_time += now - last_check
            status = "Not Focused"
            color = (0, 0, 255) 
            
        last_check = now
        
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('AI Study Focus Tracker', frame)

        
        if cv2.getWindowProperty('AI Study Focus Tracker', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()
    exit_app = True

def update_gui():
    """
    Updates the Tkinter GUI labels with the current time and status.
    """
    if not exit_app:
        
        mins_focused, secs_focused = divmod(int(focused_time), 60)
        mins_distracted, secs_distracted = divmod(int(distracted_time), 60)
        
        
        focus_label.config(text=f"Focus Time: {mins_focused:02d}:{secs_focused:02d}")
        distraction_label.config(text=f"Distraction Time: {mins_distracted:02d}:{secs_distracted:02d}")
        
        
        if timer_running and target_app:
            status_label.config(text=f"Tracking Focus on: {target_app}", fg="green")
        elif target_app:
            status_label.config(text="Not Focused", fg="red")
        else:
            status_label.config(text="Select an app to begin", fg="blue")
        
        
        root.after(1000, update_gui)
    else:
        status_label.config(text="Webcam Closed", fg="gray")

def plot_graph():
    """
    Plots a bar graph of focused vs. distracted time using Matplotlib.
    """
    labels = ['Focused', 'Distracted']
    times = [focused_time, distracted_time]
    
    
    if sum(times) > 0:
        plt.bar(labels, times)
        plt.xlabel('Status')
        plt.ylabel('Time (seconds)')
        plt.title('Focus and Distraction Time')
        plt.show()

def start_tracking_button():
    """
    Callback function for the "Start Tracking" button.
    It gets the selected app and starts the webcam thread.
    """
    global target_app
    
    selection = app_listbox.curselection()
    if selection:
        index = selection[0]
        target_app = app_listbox.get(index)
        print(f"Selected app: {target_app}")
        
        app_listbox.config(state=tk.DISABLED)
        start_button.config(state=tk.DISABLED)
        status_label.config(text=f"Tracking Focus on: {target_app}", fg="green")
    else:
        status_label.config(text="Please select an application first!", fg="red")
        
def on_close():
    """
    Handles the window close event. Stops the webcam thread and plots the graph.
    """
    global exit_app
    exit_app = True
    plot_graph()
    root.destroy()


root = tk.Tk()
root.title("AI-Powered Study Focus App")
root.geometry("400x500")
root.resizable(False, False)
root.protocol("WM_DELETE_WINDOW", on_close)


tk.Label(root, text="Select an Application to Track", font=("Arial", 16)).pack(pady=10)


app_listbox = tk.Listbox(root, height=10, width=50, selectmode=tk.SINGLE)
app_listbox.pack(pady=5)


open_windows = gw.getAllTitles()
for window in open_windows:
    if window.strip(): 
        app_listbox.insert(tk.END, window)


scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL)
scrollbar.config(command=app_listbox.yview)
app_listbox.config(yscrollcommand=scrollbar.set)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y, in_=app_listbox.master, anchor=tk.E)


start_button = tk.Button(root, text="Start Tracking", font=("Arial", 14), command=start_tracking_button)
start_button.pack(pady=10)


focus_label = tk.Label(root, text="Focus Time: 00:00", font=("Arial", 18))
focus_label.pack(pady=5)

distraction_label = tk.Label(root, text="Distraction Time: 00:00", font=("Arial", 18))
distraction_label.pack(pady=5)

status_label = tk.Label(root, text="Starting webcam...", font=("Arial", 14), fg="blue")
status_label.pack(pady=5)

info = tk.Label(root, text="Press 'q' in the webcam window to quit.\nClose this window to see the graph.", font=("Arial", 10), fg="gray")
info.pack(pady=5)

threading.Thread(target=webcam_loop, daemon=True).start()

root.after(1000, update_gui)
root.mainloop()