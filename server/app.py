from flask import Flask,render_template, request, jsonify,redirect,url_for,send_from_directory
import os
from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
from roboflow import Roboflow
from easyocr import Reader

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


def dfs_iterative(matrix, start_x, start_y, label, threshold=1):
    stack = [(start_x, start_y)]  # Use a stack to hold cells to visit
    # Directions for 8 possible moves (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while stack:
        x, y = stack.pop()  # Get the last cell added to the stack
        # Check boundaries and if the cell is part of the component
        if x < 0 or x >= matrix.shape[0] or y < 0 or y >= matrix.shape[1] or matrix[x, y] != 1:
            continue

        # Mark the cell with the label
        matrix[x, y] = label

        # Add all 8 adjacent cells to the stack
        for dx, dy in directions:
            stack.append((x + dx, y + dy))
        
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                # Ensure we don't exceed matrix boundaries
                if dx == 0 and dy == 0:
                    continue  # Skip the current cell
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < matrix.shape[0]) and (0 <= new_y < matrix.shape[1]):
                    if matrix[new_x, new_y] == 1:  # Check for connected pixels
                        stack.append((new_x, new_y))

def label_connected_components(binary_matrix, threshold=1):
    # Create a copy of the binary matrix to store labels
    labeled_matrix = binary_matrix.copy()
    # Initialize label (starting from 2 since 0 and 1 are used in the binary matrix)
    label = 2
    for i in range(labeled_matrix.shape[0]):
        for j in range(labeled_matrix.shape[1]):
            if labeled_matrix[i, j] == 1:  # Found an unvisited part of a component
                dfs_iterative(labeled_matrix, i, j, label, threshold)
                label += 1  # Increment label for the next component

    return labeled_matrix

def visualize_labels_without_compression(labeled_matrix):
    # Create a color image with the same height and width as the labeled matrix
    height, width = labeled_matrix.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a unique color map for labels
    unique_labels = np.unique(labeled_matrix)
    np.random.seed(0)  # For reproducible random colors
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))

    # Map each label to its corresponding color
    for idx, label in enumerate(unique_labels):
        if label > 1:  # Skip background (0 and 1)
            # Create a mask for the current label
            mask = (labeled_matrix == label)
            # Set the color for the pixels in the color image
            color_image[mask] = colors[idx]
    
    # Return the color image
    return color_image

def count_labels(labeled_matrix):
    unique_labels = np.unique(labeled_matrix)
    num_labels = len(unique_labels) - 1  # Exclude background (0)
    return num_labels

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

        # Converting PIL image into array
        image_array = np.array(pil_image)
        image_np=image_array

        # If the image is in RGB, convert it to BGR
        if pil_image.mode == "RGB":
            image_array = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            # For grayscale or other modes, no conversion needed
            image_array = image_array

    
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2) 

        # Define the range for blue color in HSV
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        
        #ye code ek ka hai
        # Remove blue lines by combining the masks
        final_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(blue_mask))
        output_image = np.ones_like(image_array) * 255
        # Set red areas (in the mask) to black
        output_image[final_mask > 0] = [0, 0, 0]  # Set red mask regions to black
        output_pil_image = Image.fromarray(output_image)
        output_pil_image.show()
        blackAndwhite_filename_forText = f'blackAndwhiteText_{file.filename}'
        blackAndwhite_filepath_forText = os.path.join(app.config['UPLOAD_FOLDER'],blackAndwhite_filename_forText)
        cv2.imwrite(blackAndwhite_filepath_forText, output_image)

        # Remove red lines by combining the masks
        final_mask2 = cv2.bitwise_and(blue_mask, cv2.bitwise_not(red_mask))
        output_image2 = np.ones_like(image_array) * 255
        # Set blue areas (in the mask) to black
        output_image2[final_mask2 > 0] = [0, 0, 0]  # Set blue mask regions to black
        output_pil_image2 = Image.fromarray(output_image2)
        output_pil_image2.show()

        #pehle output_image thi ab 2 kar diya
       
        blackAndwhite_filename = f'blackAndwhite_{file.filename}'
        blackAndwhite_filepath = os.path.join(app.config['UPLOAD_FOLDER'],blackAndwhite_filename)
        cv2.imwrite(blackAndwhite_filepath, output_image2)
        # cv2.imshow("asdsfdf",blackAndwhite_filepath) 
        
        # Easy OCR tasks Begin
        reader = Reader(['en'],gpu=True)
        img = Image.open(blackAndwhite_filepath_forText).convert('RGB')
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=1)
        for (bbox, text, prob) in results:
            if prob > 0.5:
                # Filter out low-confidence predictions
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                cv2.rectangle(img_array, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(img_array, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        text_detection_results=Image.fromarray(img_array)
        text_detection_results.show()    

        
        
        
        # Set confidence and overlap thresholds
        model.confidence = 50
        model.overlap = 25
        # Make a prediction
        #yahan change karna padega filepath iska dena hoga
        #shi to hai nhi dekho ek min
        prediction = model.predict(blackAndwhite_filepath)
#hmm vo pehle vaali thi

        # ismain konsi image deni hai filepath to ye hai
        image = Image.open(blackAndwhite_filepath)
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        # YOLOv5 prediction data (bounding boxes and classes)
        # Draw bounding boxes on the image ye image par ban 
        for detection in prediction:
            # Calculate the top-left and bottom-right coordinates of the bounding box
            x1 = detection["x"] - detection["width"] / 2
            y1 = detection["y"] - detection["height"] / 2
            x2 = detection["x"] + detection["width"] / 2
            y2 = detection["y"] + detection["height"] / 2

            # Draw the bounding box
            # ye hain na roboflwo waale to
            #vo doosra vaala i think ocr ka hoga
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Label the box with the class name
            font = ImageFont.truetype("arial.ttf", size=40)
            draw.text((x1, y1 - 10), detection["class"], fill="red",font=font)

        
        
        
        
        
        
        
        
        
        
        
        
        # Canny 
        canny_pil_image = Image.open(blackAndwhite_filepath)
        canny_image_array = np.array(canny_pil_image)
        gray_image = cv2.cvtColor(canny_image_array, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5,5), 1.5)
        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
        bounding_lines = np.zeros_like(edges)
        #ye vaala canny ke box hain
        for detection in prediction:
            # Calculate the top-left and bottom-right coordinates of the bounding box
            x1 = int(detection["x"] - detection["width"] / 2)
            y1 = int(detection["y"] - detection["height"] / 2)
            x2 = int(detection["x"] + detection["width"] / 2)
            y2 = int(detection["y"] + detection["height"] / 2)
            cv2.rectangle(bounding_lines, (x1, y1), (x2, y2), 255, thickness=1)
            # Set the region inside the bounding box to 0 (black) in the edges image
            edges[y1:y2, x1:x2] = 0 
        

        #canny vaale main text
        #matlab ye jo abhi kholi thi thk hai
        # Convert the edges NumPy array to a PIL Image
        edges_pil_image = Image.fromarray(edges)
        edges_pil_image.show()

        # Convert bounding_lines into PIL image
        bounding_lines_pil_image = Image.fromarray(bounding_lines)
        bounding_lines_pil_image.show()

        binary_matrix = np.where(edges > 0, 1, 0)
        # rectangle_matrix = np.where(bounding_lines>0,1,0)



        # Label the connected components with a threshold for connectivity
        threshold = 10  # Adjust the threshold as necessary
        labeled_matrix = label_connected_components(binary_matrix, threshold)
        # Visualize the labels on the image without compression
        visualized_image = visualize_labels_without_compression(labeled_matrix)
        visualized_image_pil=Image.fromarray(visualized_image)
        visualized_image_pil.show()
        num_labels = count_labels(labeled_matrix)
        image_height, image_width = image_array.shape[:2]
        # Create a blank binary matrix (label matrix) with the same dimensions as the image
        labelled_matrix_component = np.zeros((image_height, image_width), dtype=np.int32)
        # Label the boundary (outline) of each bounding box in the binary matrix with unique integer values
        for i,detection in enumerate(prediction):
            # Calculate the top-left and bottom-right coordinates of the bounding box
            x1 = int(detection["x"] - detection["width"] / 2)
            y1 = int(detection["y"] - detection["height"] / 2)
            x2 = int(detection["x"] + detection["width"] / 2)
            y2 = int(detection["y"] + detection["height"] / 2)
            # Label the boundary of the bounding box with the unique label ID (outline only)
            label_id = i+1
            cv2.rectangle(labelled_matrix_component, (x1, y1), (x2, y2), label_id, thickness=10)


        intersection_set = []
        # Step 2: Loop through the first matrix
        for i, row in enumerate(labelled_matrix_component):
            for j, value in enumerate(row):
                # Step 3: Check if the value is non-zero in both matrices
                if value != 0 and labeled_matrix[i][j] != 0:
                    # Create a key-value pair as a tuple and add to the set
                    intersection_set.append((value, labeled_matrix[i][j]))  # Correct tuple creation

        # Step 4: Display the intersection set
        # print("Intersection Set:")
        # for item in intersection_set:
            # print(item)

        no_of_component_labels = count_labels(labelled_matrix_component)
        array_of_sets = [set() for _ in range(no_of_component_labels)];
        for item in intersection_set:
            array_of_sets[item[0]-1].add(item[1])





        output_dir = './outputs'
        output_file = os.path.join(output_dir, 'output.txt')
        os.makedirs(output_dir, exist_ok=True)

        i = 0
        print(len(prediction))
        with open(output_file, 'w') as text_file:
            for detection in prediction:
                nodes = list(array_of_sets[i])
                # Calculate the top-left and bottom-right coordinates of the bounding box
                str = detection['class']
                nodes = list(array_of_sets[i])

                if(len(array_of_sets[i])==1):
                    text_file.write(f"{str} connected to only one node {array_of_sets[i].pop()}\n")
                elif(len(nodes)>=2):
                    text_file.write(f"{str} connected to node {nodes[0]} and {nodes[1]}\n")
                i += 1
        









       

































        # Save processed image
        processed_filename = f'processed_{file.filename}'
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'],processed_filename)
        image.save(processed_path)

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