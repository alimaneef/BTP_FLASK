from flask import Flask,render_template, request, jsonify,redirect,url_for,send_from_directory
import os
from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
from roboflow import Roboflow

app=Flask(__name__,template_folder='templates')

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    elif request.method=='POST':
        username=request.form.get('username')
        password=request.form.get('password')

        if(username=='India' and password=='pass'):
            return "Success"
        else:
            return "Failed"
        

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Initialize Roboflow
rf = Roboflow(api_key="s5x7SRtXClTYVdRddG1E")
project = rf.workspace().project("electronic-circuit-iysi9")
model = project.version("3").model

# Helper function to check file extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image part in the request.'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected for uploading.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Unsupported file type. Allowed types: PNG, JPG, JPEG, GIF.'}), 400

    # Save the file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Open and process the image
    try:
        # Open the image with Pillow
        pil_image = Image.open(filepath)

        # Convert the image to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        



















        # Example processing: Convert to grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Set confidence and overlap thresholds
        model.confidence = 50
        model.overlap = 25

        # Make a prediction
        prediction = model.predict(filepath)




        image = Image.open(filepath)

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # YOLOv5 prediction data (bounding boxes and classes)

        # Draw bounding boxes on the image
        for detection in prediction:
            # Calculate the top-left and bottom-right coordinates of the bounding box
            x1 = detection["x"] - detection["width"] / 2
            y1 = detection["y"] - detection["height"] / 2
            x2 = detection["x"] + detection["width"] / 2
            y2 = detection["y"] + detection["height"] / 2

            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Label the box with the class name
            font = ImageFont.truetype("arial.ttf", size=40)
            draw.text((x1, y1 - 10), detection["class"], fill="red",font=font)





















        # Save processed image
        processed_filename = f'processed_{file.filename}'
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'],processed_filename )
        image.save(processed_path)
        # cv2.imwrite(processed_path, gray_image)

        # return jsonify({'success': True, 'message': 'Image uploaded and processed successfully!', 'processed_image': processed_path})
        return redirect(url_for('show_result', filename=processed_filename))
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'}), 500

@app.route('/result/<filename>')
def show_result(filename):
    processed_image_url = url_for('uploaded_file', filename=filename)
    return render_template('result.html', image_url=processed_image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__== '__main__':
    app.run(host='0.0.0.0',debug=True)