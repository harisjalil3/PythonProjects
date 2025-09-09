import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import cv2
import time
import threading
import pygetwindow as gw
import mediapipe as mp
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import io
import collections

# --- Global Variables ---
focused_time = 0
distracted_time = 0
target_app_titles = []  # Changed to a list for multiple app titles
tracking_active = False
face_detected_in_frame = False
head_facing_forward = False
# Variables for video capture and frame handling
cap = None
frame_buffer = []
# Locks for thread-safe access
lock = threading.Lock()
stop_event = threading.Event()
# Lists to store data for the graph
focus_data = [] # Stores tuples of (timestamp, focused_time)
distraction_data = [] # Stores tuples of (timestamp, distracted_time)
# Dictionary to store distraction time per application
distraction_per_app = collections.defaultdict(float)
start_time = time.time()
# Pomodoro timer variables
pomodoro_minutes = 25
pomodoro_seconds = 0
pomodoro_running = False
pomodoro_paused = False
pomodoro_thread = None
# Gamification variables
focus_session_count = 0
# Variables for dynamic video display
last_frame = None

# --- Helper Functions ---
def get_active_window_title():
    """Gets the title of the currently active window."""
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            return active_window.title
        else:
            return ""
    except gw.PyGetWindowException:
        return ""

def list_windows():
    """Populates the listbox with titles of open windows."""
    app_listbox.delete(0, tk.END)
    try:
        open_windows = [title for title in gw.getAllTitles() if title.strip()]
        if not open_windows:
            messagebox.showinfo("No Windows", "No open windows found. Please open an application to track.")
            return
        for title in open_windows:
            app_listbox.insert(tk.END, title)
    except Exception as e:
        messagebox.showerror("Error", f"Could not list open windows: {e}")

def get_normalized_app_name(window_title):
    """
    Normalizes a window title to a common application name.
    This helps group time for different windows of the same application.
    """
    title_lower = window_title.strip().lower()
    
    if "chrome" in title_lower:
        return "Google Chrome"
    elif "firefox" in title_lower:
        return "Mozilla Firefox"
    elif "code" in title_lower:
        return "Visual Studio Code"
    elif "word" in title_lower:
        return "Microsoft Word"
    elif "excel" in title_lower:
        return "Microsoft Excel"
    elif "powerpoint" in title_lower:
        return "Microsoft PowerPoint"
    elif "file explorer" in title_lower:
        return "File Explorer"
    elif "spotify" in title_lower:
        return "Spotify"
    elif "terminal" in title_lower or "cmd" in title_lower or "powershell" in title_lower:
        return "Terminal"
    # Add more common applications here
    
    # Return the original title if no match is found
    return window_title

# --- Pomodoro Timer Functions ---
def pomodoro_timer_loop():
    """Handles the countdown for the Pomodoro timer, pausing on distraction."""
    global pomodoro_minutes, pomodoro_seconds, pomodoro_running, pomodoro_paused, focus_session_count, tracking_active, face_detected_in_frame, head_facing_forward, target_app_titles
    
    while pomodoro_running:
        # Check for distraction and pause the timer
        current_active_window = get_active_window_title()
        is_focused_on_app = current_active_window.strip().lower() in [t.strip().lower() for t in target_app_titles]
        is_focused = tracking_active and head_facing_forward and face_detected_in_frame and is_focused_on_app
        
        if not is_focused:
            pomodoro_paused = True
            pomodoro_label.config(text=f"Timer Paused: {pomodoro_minutes:02d}:{pomodoro_seconds:02d}")
            time.sleep(1) # Sleep to prevent busy-waiting
            continue
        
        pomodoro_paused = False
        pomodoro_label.config(text=f"Time Left: {pomodoro_minutes:02d}:{pomodoro_seconds:02d}")

        if pomodoro_minutes == 0 and pomodoro_seconds == 0:
            pomodoro_running = False
            focus_session_count += 1
            gamification_label.config(text=f"🎉 You completed a focus session! Total sessions: {focus_session_count} 🎉")
            messagebox.showinfo("Pomodoro", "Time to take a break! 🥳")
            reset_pomodoro()
            return

        time.sleep(1)
        if pomodoro_seconds > 0:
            pomodoro_seconds -= 1
        else:
            pomodoro_seconds = 59
            pomodoro_minutes -= 1

def start_pomodoro():
    """Starts the Pomodoro timer."""
    global pomodoro_running, pomodoro_thread, pomodoro_paused
    if not pomodoro_running:
        pomodoro_running = True
        pomodoro_paused = False
        pomodoro_thread = threading.Thread(target=pomodoro_timer_loop, daemon=True)
        pomodoro_thread.start()
        pomodoro_start_button.config(state=tk.DISABLED)
        pomodoro_pause_button.config(state=tk.NORMAL)
        pomodoro_stop_button.config(state=tk.NORMAL)

def pause_pomodoro():
    """Pauses or resumes the Pomodoro timer."""
    global pomodoro_paused
    pomodoro_paused = not pomodoro_paused
    if pomodoro_paused:
        pomodoro_pause_button.config(text="Resume")
    else:
        pomodoro_pause_button.config(text="Pause")
        # Restart the loop to continue the timer
        threading.Thread(target=pomodoro_timer_loop, daemon=True).start()

def reset_pomodoro():
    """Stops and resets the Pomodoro timer."""
    global pomodoro_running, pomodoro_minutes, pomodoro_seconds, pomodoro_paused
    pomodoro_running = False
    pomodoro_paused = False
    pomodoro_minutes = 25
    pomodoro_seconds = 0
    pomodoro_label.config(text=f"Time Left: {pomodoro_minutes:02d}:{pomodoro_seconds:02d}")
    pomodoro_start_button.config(state=tk.NORMAL)
    pomodoro_pause_button.config(state=tk.DISABLED, text="Pause")
    pomodoro_stop_button.config(state=tk.DISABLED)

# --- Core Logic Functions ---
def webcam_loop():
    """Background thread for webcam and face tracking."""
    global focused_time, distracted_time, tracking_active, face_detected_in_frame, head_facing_forward, cap, frame_buffer, focus_data, distraction_data, last_frame, start_time, distraction_per_app, target_app_titles
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        stop_event.set()
        return

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        last_check = time.time()
        start_time = time.time()
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            # Process the frame for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            with lock:
                face_detected_in_frame = bool(results.multi_face_landmarks)
                
            if face_detected_in_frame:
                face_landmarks = results.multi_face_landmarks[0]
                # A simple check for head orientation: if the nose tip's x-coordinate is too far from the center
                image_width, image_height = frame.shape[1], frame.shape[0]
                nose_tip = face_landmarks.landmark[1] # Landmark for the nose tip
                x_normalized = nose_tip.x
                center_x = 0.5
                # Define a threshold for "forward-facing"
                if abs(x_normalized - center_x) < 0.15: # 0.15 is a simple threshold
                    head_facing_forward = True
                else:
                    head_facing_forward = False
            else:
                head_facing_forward = False
            
            now = time.time()
            current_active_window = get_active_window_title()

            with lock:
                is_focused_on_app = current_active_window.strip().lower() in [t.strip().lower() for t in target_app_titles]
                is_focused = tracking_active and face_detected_in_frame and head_facing_forward and is_focused_on_app
                
                delta_time = now - last_check
                if is_focused:
                    focused_time += delta_time
                else:
                    distracted_time += delta_time
                    # --- FIX: Use the normalized app name for the dictionary key ---
                    normalized_app_name = get_normalized_app_name(current_active_window)
                    if normalized_app_name:
                        distraction_per_app[normalized_app_name] += delta_time
                    else:
                        distraction_per_app["(No Active Window)"] += delta_time
                last_check = now
                
                # Update data for the graph every second
                if int(now - start_time) > len(focus_data) + 1:
                    focus_data.append((now - start_time, focused_time))
                    distraction_data.append((now - start_time, distracted_time))
            
            # Draw landmarks on the frame
            if face_detected_in_frame:
                mp_drawing = mp.solutions.drawing_utils
                for face_landmarks in results.multi_face_landmarks:
                    # Draw landmarks with a different color if distracted
                    drawing_spec_color = (0, 255, 0) if is_focused else (0, 0, 255)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=drawing_spec_color, thickness=1, circle_radius=1))
            
            # Store the frame in a buffer for the GUI to display
            with lock:
                last_frame = frame.copy()

    cap.release()
    print("Webcam loop stopped.")

def update_gui():
    """Updates the GUI with new data from the webcam thread."""
    global focused_time, distracted_time, tracking_active, face_detected_in_frame, head_facing_forward, last_frame
    
    if last_frame is not None:
        frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frame)
        
        # Resize the image to fit the label
        width = video_label.winfo_width()
        height = video_label.winfo_height()
        if width > 0 and height > 0:
            img = img.resize((width, height))
        
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.config(image=img_tk)
        video_label.image = img_tk
    
    # Update stats labels
    focused_label.config(text=f"Focused Time: {int(focused_time)}s")
    distraction_label.config(text=f"Distraction Time: {int(distracted_time)}s")
    
    # Update status message based on new granular tracking
    current_active_window = get_active_window_title()
    if tracking_active:
        if not face_detected_in_frame:
            status_message = "Status: User is not detected. The PC is idle."
        elif not head_facing_forward:
            status_message = "Status: Distracted (looking away)"
        elif current_active_window.strip().lower() not in [t.strip().lower() for t in target_app_titles]:
            status_message = f"Status: Distracted (on '{current_active_window}')"
        else:
            status_message = f"Status: Focusing on selected applications"
    else:
        status_message = "Status: Idle"
    status_label.config(text=status_message)

    # Schedule the next update
    root.after(10, update_gui)

def start_tracking_button_handler():
    """Starts the tracking process."""
    global focused_time, distracted_time, target_app_titles, tracking_active, stop_event, focus_data, distraction_data, start_time, distraction_per_app
    
    selected_indices = app_listbox.curselection()
    if not selected_indices:
        messagebox.showerror("No Selection", "Please select at least one application to track.")
        return

    # Get all selected application titles
    target_app_titles = [app_listbox.get(i) for i in selected_indices]
    
    with lock:
        focused_time = 0
        distracted_time = 0
        focus_data.clear()
        distraction_data.clear()
        distraction_per_app.clear() # Clear the distraction report data
        start_time = time.time()
        tracking_active = True
        stop_event.clear()
        
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    show_graph_button.config(state=tk.DISABLED)
    show_report_button.config(state=tk.DISABLED) # Disable report button
    refresh_button.config(state=tk.DISABLED)
    gamification_label.config(text="Tracking started. Get ready to focus!")

    # Start the webcam thread
    threading.Thread(target=webcam_loop, daemon=True).start()

def stop_tracking_button_handler():
    """Stops the tracking process."""
    global tracking_active, stop_event
    with lock:
        tracking_active = False
        stop_event.set()
    
    # Give a moment for the thread to stop
    root.after(500, finalize_stop)

def finalize_stop():
    """Finalizes the GUI after the tracking thread has stopped."""
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    show_graph_button.config(state=tk.NORMAL)
    show_report_button.config(state=tk.NORMAL) # Enable report button
    refresh_button.config(state=tk.NORMAL)
    gamification_label.config(text="Tracking stopped. Check your stats!")

def show_graph():
    """Displays a graph of focus vs. distraction time."""
    if len(focus_data) < 2 or len(distraction_data) < 2:
        messagebox.showinfo("No Data", "Not enough tracking data to plot a line graph.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Unpack the data for plotting
    focus_x = [d[0] for d in focus_data]
    focus_y = [d[1] for d in focus_data]
    distraction_x = [d[0] for d in distraction_data]
    distraction_y = [d[1] for d in distraction_data]

    ax.plot(focus_x, focus_y, label='Focused Time', color='#4CAF50')
    ax.plot(distraction_x, distraction_y, label='Distraction Time', color='#FF5733')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Cumulative Time (seconds)')
    ax.set_title('Focus vs. Distraction Over Time')
    ax.legend()
    ax.grid(True)
    
    # Open a new Tkinter window for the plot
    graph_window = tk.Toplevel(root)
    graph_window.title("Focus Analytics")
    
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_distraction_report():
    """Displays a detailed report of distraction time by application."""
    global distraction_per_app
    if not distraction_per_app:
        messagebox.showinfo("No Data", "No distraction data was recorded.")
        return

    # Create a new window for the report
    report_window = tk.Toplevel(root)
    report_window.title("Distraction Report")
    
    report_frame = ttk.Frame(report_window, padding=10)
    report_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(report_frame, text="Distraction Time by Application", font=('Helvetica', 16, 'bold')).pack(pady=5)
    
    # Sort the dictionary by distraction time in descending order
    sorted_distractions = sorted(distraction_per_app.items(), key=lambda item: item[1], reverse=True)
    
    # Create a Text widget with a scrollbar
    report_text = tk.Text(report_frame, wrap=tk.WORD, font=('Helvetica', 12), state='disabled', height=20, width=60)
    report_scrollbar = ttk.Scrollbar(report_frame, command=report_text.yview)
    report_text.configure(yscrollcommand=report_scrollbar.set)
    
    report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    report_text.config(state='normal') # Enable editing to insert text
    report_text.insert(tk.END, "Apps are ranked by the total time spent distracted on them.\n\n")

    for app_title, time_spent in sorted_distractions:
        report_text.insert(tk.END, f"  - {app_title}: {time_spent:.2f} seconds\n")
        
    report_text.config(state='disabled') # Disable editing again

def on_closing():
    """Handles the window closing event to ensure cleanup."""
    if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
        stop_tracking_button_handler()
        root.destroy()

def main():
    """The main function to set up and run the GUI."""
    global root, video_label, app_listbox, start_button, stop_button, show_graph_button, refresh_button, status_label, focused_label, distraction_label, pomodoro_label, pomodoro_start_button, pomodoro_stop_button, pomodoro_pause_button, gamification_label, show_report_button
    
    root = tk.Tk()
    root.title("AI Focus Tracker")
    root.state('zoomed')
    root.protocol("WM_DELETE_WINDOW", on_closing)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 12))
    style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=10)

    # Main frame using a grid for better control
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)
    
    # App Selection Frame (Top-left)
    app_selection_frame = ttk.LabelFrame(main_frame, text="app selection", padding=10)
    app_selection_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
    app_selection_frame.rowconfigure(1, weight=1)
    app_selection_frame.columnconfigure(0, weight=1)
    
    # Add selectmode=tk.MULTIPLE to allow multiple selections
    app_listbox = tk.Listbox(app_selection_frame, height=10, width=40, selectmode=tk.MULTIPLE)
    app_listbox.grid(row=1, column=0, columnspan=2, pady=5, sticky='nsew')
    
    # App Selection Buttons (Bottom of app selection frame)
    app_buttons_frame = ttk.Frame(app_selection_frame)
    app_buttons_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky='nsew')
    app_buttons_frame.columnconfigure(0, weight=1)
    app_buttons_frame.columnconfigure(1, weight=1)
    app_buttons_frame.columnconfigure(2, weight=1)
    refresh_button = ttk.Button(app_buttons_frame, text="Refresh", command=list_windows)
    refresh_button.grid(row=0, column=0, padx=5, sticky='nsew')
    start_button = ttk.Button(app_buttons_frame, text="Start Tracking", command=start_tracking_button_handler)
    start_button.grid(row=0, column=1, padx=5, sticky='nsew')
    stop_button = ttk.Button(app_buttons_frame, text="Stop Tracking", command=stop_tracking_button_handler, state=tk.DISABLED)
    stop_button.grid(row=0, column=2, padx=5, sticky='nsew')
    
    # Graph and Report buttons (Below the small buttons)
    show_graph_button = ttk.Button(app_selection_frame, text="Show Stats Graph", command=show_graph, state=tk.DISABLED)
    show_graph_button.grid(row=3, column=0, columnspan=1, pady=10, sticky='nsew')
    show_report_button = ttk.Button(app_selection_frame, text="Show Distraction Report", command=show_distraction_report, state=tk.DISABLED)
    show_report_button.grid(row=3, column=1, columnspan=1, pady=10, sticky='nsew')

    # Live Webcam Frame (Top-right)
    video_frame = ttk.LabelFrame(main_frame, text="live webcam", padding=10)
    video_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
    video_frame.columnconfigure(0, weight=1)
    video_frame.rowconfigure(0, weight=1)
    video_label = ttk.Label(video_frame)
    video_label.grid(row=0, column=0, sticky='nsew')

    # Pomodoro Timer Frame (Bottom-right)
    pomodoro_frame = ttk.LabelFrame(main_frame, text="pomodoro timer", padding=10)
    pomodoro_frame.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')
    pomodoro_frame.columnconfigure(0, weight=1)
    pomodoro_frame.rowconfigure(0, weight=1)
    pomodoro_label = ttk.Label(pomodoro_frame, text="Time Left: 25:00", font=('Helvetica', 24, 'bold'))
    pomodoro_label.grid(row=0, column=0, columnspan=3, pady=10, sticky='nsew')

    # Pomodoro Buttons
    pomodoro_buttons_frame = ttk.Frame(pomodoro_frame)
    pomodoro_buttons_frame.grid(row=1, column=0, columnspan=3, sticky='nsew')
    pomodoro_buttons_frame.columnconfigure(0, weight=1)
    pomodoro_buttons_frame.columnconfigure(1, weight=1)
    pomodoro_buttons_frame.columnconfigure(2, weight=1)
    pomodoro_start_button = ttk.Button(pomodoro_buttons_frame, text="Start", command=start_pomodoro)
    pomodoro_start_button.grid(row=0, column=0, padx=5, sticky='nsew')
    pomodoro_pause_button = ttk.Button(pomodoro_buttons_frame, text="Pause", command=pause_pomodoro, state=tk.DISABLED)
    pomodoro_pause_button.grid(row=0, column=1, padx=5, sticky='nsew')
    pomodoro_stop_button = ttk.Button(pomodoro_buttons_frame, text="Reset", command=reset_pomodoro, state=tk.DISABLED)
    pomodoro_stop_button.grid(row=0, column=2, padx=5, sticky='nsew')
    
    # Status and Time Display (Bottom-left)
    stats_frame = ttk.Frame(main_frame)
    stats_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
    status_label = ttk.Label(stats_frame, text="Status: Idle", font=('Helvetica', 14, 'italic'))
    status_label.pack(pady=10, anchor='w')
    focused_label = ttk.Label(stats_frame, text="Focused Time: 0s", font=('Helvetica', 14))
    focused_label.pack(pady=5, anchor='w')
    distraction_label = ttk.Label(stats_frame, text="Distraction Time: 0s", font=('Helvetica', 14))
    distraction_label.pack(pady=5, anchor='w')
    gamification_label = ttk.Label(stats_frame, text="", font=('Helvetica', 14, 'bold'), foreground='blue')
    gamification_label.pack(pady=10, anchor='w')
    
    list_windows()
    update_gui()
    root.mainloop()

if __name__ == '__main__':
    main()
