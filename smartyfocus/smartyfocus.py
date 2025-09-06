import os
import io
import cv2
import time
import threading
import pygetwindow as gw
import matplotlib.pyplot as plt
import mediapipe as mp
from flask import Flask, render_template_string, Response, jsonify

# You will need to install the following libraries:
# pip install Flask opencv-python mediapipe pygetwindow matplotlib

# Flask app initialization
app = Flask(__name__)

# Global variables for state management
focused_time = 0
distracted_time = 0
timer_running = False
target_app = ""
webcam_active = False

# Threading events and locks
lock = threading.Lock()
stop_event = threading.Event()

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# HTML for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Web Focus Tracker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
    <div class="bg-white p-8 rounded-2xl shadow-xl w-full max-w-2xl">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-4">AI Web Focus Tracker</h1>
        <p class="text-gray-600 text-center mb-6">
            Track your focus on a selected application using your webcam.
        </p>

        <!-- Video Stream and Info -->
        <div class="flex flex-col items-center mb-6">
            <div class="relative w-full rounded-xl overflow-hidden shadow-lg">
                <img id="video_feed" src="{{ url_for('video_feed') }}" class="w-full rounded-xl"/>
                <div class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white text-lg font-semibold" id="loading-message">
                    Loading webcam...
                </div>
            </div>
            <div id="status_message" class="mt-4 text-lg font-medium text-blue-600">
                Tracking not started.
            </div>
        </div>

        <!-- Application Selection and Controls -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="bg-gray-50 p-4 rounded-xl">
                <label for="app_listbox" class="block text-gray-700 font-medium mb-2">Select Application:</label>
                <select id="app_listbox" size="8" class="w-full rounded-lg border border-gray-300 p-2 focus:ring-indigo-500 focus:border-indigo-500"></select>
            </div>
            <div class="flex flex-col space-y-4">
                <button id="start_button" onclick="startTracking()" class="w-full bg-indigo-600 text-white py-3 px-4 rounded-full font-semibold hover:bg-indigo-700 transition-transform transform hover:scale-105 shadow-lg">
                    Start Tracking
                </button>
                <button id="stop_button" onclick="stopTracking()" class="w-full bg-red-600 text-white py-3 px-4 rounded-full font-semibold hover:bg-red-700 transition-transform transform hover:scale-105 shadow-lg">
                    Stop Tracking
                </button>
                <button id="show_graph_button" onclick="showGraph()" class="w-full bg-green-600 text-white py-3 px-4 rounded-full font-semibold hover:bg-green-700 transition-transform transform hover:scale-105 shadow-lg">
                    Show Focus Graph
                </button>
            </div>
        </div>

        <!-- Time Display -->
        <div class="mt-6 text-center">
            <div class="bg-indigo-100 p-4 rounded-xl shadow-inner">
                <p id="focus_time_label" class="text-xl font-bold text-indigo-800">Focus Time: 00:00</p>
            </div>
            <div class="mt-4 bg-red-100 p-4 rounded-xl shadow-inner">
                <p id="distraction_time_label" class="text-xl font-bold text-red-800">Distraction Time: 00:00</p>
            </div>
        </div>
    </div>

    <!-- Modal for showing graph -->
    <div id="graph_modal" class="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center hidden">
        <div class="bg-white p-6 rounded-lg shadow-2xl relative">
            <h2 class="text-xl font-bold mb-4">Focus and Distraction Time</h2>
            <img id="graph_image" src="" alt="Focus Graph" class="rounded-lg"/>
            <button onclick="closeModal()" class="absolute top-2 right-2 text-gray-500 hover:text-gray-800 text-2xl font-bold">&times;</button>
        </div>
    </div>

    <script>
        const appListbox = document.getElementById('app_listbox');
        const startButton = document.getElementById('start_button');
        const stopButton = document.getElementById('stop_button');
        const focusTimeLabel = document.getElementById('focus_time_label');
        const distractionTimeLabel = document.getElementById('distraction_time_label');
        const statusMessage = document.getElementById('status_message');
        const videoFeed = document.getElementById('video_feed');
        const loadingMessage = document.getElementById('loading-message');
        const graphModal = document.getElementById('graph_modal');
        const graphImage = document.getElementById('graph_image');

        stopButton.disabled = true;

        async function getOpenWindows() {
            try {
                const response = await fetch('/get_windows');
                const data = await response.json();
                appListbox.innerHTML = '';
                data.windows.forEach(window => {
                    const option = document.createElement('option');
                    option.textContent = window;
                    option.value = window;
                    appListbox.appendChild(option);
                });
            } catch (error) {
                console.error("Failed to fetch open windows:", error);
            }
        }

        async function updateStats() {
            try {
                const response = await fetch('/get_stats');
                const data = await response.json();
                
                const formatTime = (seconds) => {
                    const mins = Math.floor(seconds / 60);
                    const secs = Math.floor(seconds % 60);
                    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
                };

                focusTimeLabel.textContent = `Focus Time: ${formatTime(data.focused_time)}`;
                distractionTimeLabel.textContent = `Distraction Time: ${formatTime(data.distracted_time)}`;
                
                if (data.tracking) {
                    statusMessage.textContent = `Tracking Focus on: ${data.target_app}`;
                    statusMessage.style.color = 'green';
                } else if (data.target_app) {
                    statusMessage.textContent = 'Not Focused';
                    statusMessage.style.color = 'red';
                }
            } catch (error) {
                console.error("Failed to fetch stats:", error);
            }
            setTimeout(updateStats, 1000);
        }

        async function startTracking() {
            const selectedApp = appListbox.value;
            if (!selectedApp) {
                statusMessage.textContent = "Please select an application first!";
                statusMessage.style.color = 'red';
                return;
            }

            try {
                const response = await fetch('/start_tracking', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ app_name: selectedApp })
                });
                const data = await response.json();
                if (data.success) {
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    appListbox.disabled = true;
                } else {
                    statusMessage.textContent = "Failed to start tracking.";
                    statusMessage.style.color = 'red';
                }
            } catch (error) {
                console.error("Failed to start tracking:", error);
            }
        }

        async function stopTracking() {
            try {
                const response = await fetch('/stop_tracking', { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    appListbox.disabled = false;
                    statusMessage.textContent = "Tracking stopped.";
                    statusMessage.style.color = 'blue';
                }
            } catch (error) {
                console.error("Failed to stop tracking:", error);
            }
        }

        function showGraph() {
            graphImage.src = "{{ url_for('plot_graph') }}?" + new Date().getTime(); // Prevent caching
            graphModal.classList.remove('hidden');
        }

        function closeModal() {
            graphModal.classList.add('hidden');
        }
        
        videoFeed.onload = () => {
            loadingMessage.classList.add('hidden');
        };

        window.onload = () => {
            getOpenWindows();
            updateStats();
        };

    </script>
</body>
</html>
"""

def get_active_window_title():
    """
    Gets the title of the currently active window.
    """
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            return active_window.title
        else:
            return ""
    except gw.PyGetWindowException:
        return ""

def is_face_detected(frame):
    """
    Uses MediaPipe to detect a face in the given frame.
    Returns True if a face is found, False otherwise.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    return bool(results.detections)

def webcam_loop():
    """
    Main loop to capture video, detect face, and track focus time.
    Runs in a separate thread.
    """
    global focused_time, distracted_time, timer_running
    cap = cv2.VideoCapture(0)
    last_check = time.time()
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        face_detected = is_face_detected(frame)
        current_active_window = get_active_window_title()

        with lock:
            now = time.time()
            if target_app and face_detected and (current_active_window == target_app):
                timer_running = True
                focused_time += now - last_check
            else:
                timer_running = False
                distracted_time += now - last_check
            last_check = now

        # Encode frame for video feed
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    stop_event.set()

@app.route('/')
def index():
    """Renders the main web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_windows')
def get_windows():
    """Returns a JSON list of open application window titles."""
    try:
        open_windows = [title for title in gw.getAllTitles() if title.strip()]
        return jsonify(windows=open_windows)
    except Exception as e:
        print(f"Error getting windows: {e}")
        return jsonify(windows=[]), 500

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    """Starts the focus tracking process."""
    global target_app
    data = request.json
    target_app = data.get('app_name')
    if target_app:
        return jsonify(success=True)
    return jsonify(success=False), 400

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    """Stops the focus tracking process."""
    global target_app
    target_app = ""
    return jsonify(success=True)

@app.route('/get_stats')
def get_stats():
    """Returns the current focused and distracted times."""
    with lock:
        return jsonify(
            focused_time=focused_time,
            distracted_time=distracted_time,
            tracking=timer_running,
            target_app=target_app
        )

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Feeds the webcam frames."""
    global webcam_active
    if not webcam_active:
        threading.Thread(target=webcam_loop, daemon=True).start()
        webcam_active = True
    return Response(webcam_loop(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plot_graph')
def plot_graph():
    """Generates and returns a bar graph of focused vs. distracted time."""
    labels = ['Focused', 'Distracted']
    times = [focused_time, distracted_time]
    
    if sum(times) > 0:
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots()
        ax.bar(labels, times, color=['#4F46E5', '#EF4444'])
        ax.set_xlabel('Status')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Focus and Distraction Time')
        
        # Save plot to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Response(buf.read(), mimetype='image/png')
    
    return "No data to plot.", 404

def open_browser():
    """This function opens the default web browser to the app's URL."""
    webbrowser.open_new_tab('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Start a timer to open the browser after a 1-second delay
    Timer(1, open_browser).start()
    # Run the Flask application
    app.run(debug=False)
