from flask import Flask, request, render_template, jsonify, send_from_directory
import os

from predict import image_to_text

app = Flask(__name__)

UPLOADED_FOLDER = 'uploaded_files'
app.config['UPLOADED_FOLDER'] = UPLOADED_FOLDER
STATIC_FOLDER = 'static/images'

if not os.path.exists(UPLOADED_FOLDER):
    os.makedirs(UPLOADED_FOLDER)

@app.route('/', methods=['GET'])
def upload_form():
    sample_images = os.listdir(STATIC_FOLDER)
    return render_template('home.html', images=sample_images)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOADED_FOLDER'], file.filename)
        text = image_to_text(filepath)
        # file.save(filepath)
        # f = open('predicted_text.txt', encoding='utf-8')
        # text = f.read()
        # f.close()
        return jsonify({'text': text})
    else:
        return "File not allowed"
    
@app.route('/upload-sample', methods=['POST'])
def upload_sample_image():
    filename = request.form.get('filename')
    if filename and allowed_file(filename):
        source_path = os.path.join(STATIC_FOLDER, filename)
        dest_path = os.path.join(app.config['UPLOADED_FOLDER'], filename)
        text = image_to_text(source_path)
        # predicted_text = ''.join(text)
        # print('pt ', text)
        # f = open('predicted_text.txt', encoding='utf-8')
        # text = f.read()
        # f.close()
        return jsonify({'text': text})
    return "File not allowed or not found"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
