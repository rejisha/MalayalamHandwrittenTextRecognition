from flask import Flask, request, render_template, jsonify, send_from_directory
from flask import send_file
import os

from predict import image_to_text

app = Flask(__name__)

UPLOADED_FOLDER = 'uploaded_files'
STATIC_FOLDER = 'static/images'
TEXT_FOLDER = 'predicted_texts' 
app.config['UPLOADED_FOLDER'] = UPLOADED_FOLDER
app.config['TEXT_FOLDER'] = TEXT_FOLDER

if not os.path.exists(UPLOADED_FOLDER):
    os.makedirs(UPLOADED_FOLDER)

if not os.path.exists(TEXT_FOLDER):
    os.makedirs(TEXT_FOLDER)

@app.route('/', methods=['GET'])
def upload_form():
    sample_images = os.listdir(STATIC_FOLDER)
    return render_template('home.html', images=sample_images)

@app.route('/upload', methods=['POST'])
def image_upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':  # Change here from file.file_name to file.filename
        return "No selected file"
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOADED_FOLDER'], file.filename)
        file.save(file_path)  # Save the file
        text = image_to_text(file_path)  # This should return the text from the image
        predicted_text = ''.join(text)
        
        text_file_path = os.path.join(app.config['TEXT_FOLDER'], 'predicted_text.txt')
        # print('tf ', text_file_path)
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(predicted_text)
        return jsonify({'text': text, 'file_name': file.filename, 'text_file': 'predicted_text.txt'})
    else:
        return "File not allowed"

@app.route('/download-text/<filename>')
def download_text(filename):
    file_path = os.path.join(app.config['TEXT_FOLDER'], filename)
    print(file_path)
    return send_file(file_path, as_attachment=True)

# Ensure you have a route to serve the uploaded images:
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADED_FOLDER'], filename)

    
@app.route('/upload-sample', methods=['POST'])
def upload_sample_image():
    filename = request.form.get('filename')
    if filename and allowed_file(filename):
        source_path = os.path.join(STATIC_FOLDER, filename)
        dest_path = os.path.join(app.config['UPLOADED_FOLDER'], filename)
        text = image_to_text(source_path)
        predicted_text = ''.join(text)
        text_file_path = os.path.join(app.config['TEXT_FOLDER'], 'predicted_text.txt')
        # print('tf ', text_file_path)
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(predicted_text)
        return jsonify({'text': text, 'text_file': 'predicted_text.txt'})
    else:
        return "File not allowed"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
