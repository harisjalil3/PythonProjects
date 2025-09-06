import os
import io
import zipfile
import webbrowser
from threading import Timer
from flask import Flask, render_template_string, request, send_file
import PyPDF2
from werkzeug.utils import secure_filename
from PIL import Image

# You will need to install Flask, PyPDF2, and Pillow:
# pip install Flask PyPDF2 Pillow

# Configure the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max file size for multiple images

# HTML for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image to PDFs Converter</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
    </style>
</head>
<body class="bg-gray-100 p-8">
    <div class="bg-white p-8 rounded-2xl shadow-xl max-w-lg w-full text-center">
        <h1 class="text-3xl font-bold text-gray-800 mb-4">Image to PDFs Converter</h1>
        <p class="text-gray-600 mb-6">Upload one or more images to convert each into a separate PDF file.</p>
        <form action="/convert_images" method="post" enctype="multipart/form-data" class="space-y-4">
            <div class="flex flex-col items-center justify-center">
                <label for="images" class="block text-gray-700 font-medium mb-2">Select Image Files:</label>
                <input type="file" name="images" id="images" accept="image/*" multiple
                    class="block w-full text-sm text-gray-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-full file:border-0
                    file:text-sm file:font-semibold
                    file:bg-indigo-50 file:text-indigo-700
                    hover:file:bg-indigo-100" required>
            </div>
            <button type="submit" class="w-full bg-indigo-600 text-white py-2 px-4 rounded-full font-semibold
                hover:bg-indigo-700 transition-colors duration-300 transform hover:scale-105 shadow-lg">
                Convert to PDFs
            </button>
        </form>
        {% if message %}
        <div class="mt-6 p-4 rounded-lg {{ 'bg-red-100 text-red-700' if 'Error' in message else 'bg-green-100 text-green-700' }}">
            <p>{{ message }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Renders the main page with the file upload form."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/convert_images', methods=['POST'])
def convert_images_to_pdf():
    """Handles the image file uploads and conversion process."""
    if 'images' not in request.files:
        return render_template_string(HTML_TEMPLATE, message="Error: No image files part in the request.")

    files = request.files.getlist('images')
    if not files or all(file.filename == '' for file in files):
        return render_template_string(HTML_TEMPLATE, message="Error: No selected files.")

    images_to_convert = []
    
    for file in files:
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                img = Image.open(file)
                img = img.convert('RGB')
                images_to_convert.append((img, file.filename))
            except Exception as e:
                print(f"Failed to process image {file.filename}: {e}")

    if not images_to_convert:
        return render_template_string(HTML_TEMPLATE, message="Error: No valid image files were uploaded.")

    try:
        memory_zip = io.BytesIO()
        with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img, filename in images_to_convert:
                pdf_output = io.BytesIO()
                
                # Save each image to a separate PDF in memory
                img.save(pdf_output, "PDF", resolution=100.0)
                pdf_output.seek(0)
                
                # Create a filename for the PDF
                base_name = os.path.splitext(secure_filename(filename))[0]
                pdf_filename = f'{base_name}.pdf'
                
                # Write the PDF to the in-memory zip file
                zf.writestr(pdf_filename, pdf_output.read())

        memory_zip.seek(0)

        # Send the created zip file to the user for download
        return send_file(
            memory_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name='converted_images.zip'
        )

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, message=f"An unexpected error occurred during PDF creation: {e}")

def open_browser():
    """This function opens the default web browser to the app's URL."""
    webbrowser.open_new_tab('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Start a timer to open the browser after a 1-second delay
    Timer(1, open_browser).start()
    # Run the Flask application
    app.run(debug=False)
