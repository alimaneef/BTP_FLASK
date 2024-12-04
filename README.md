
# Electronics Project
This project allows you to upload an image, process it to detect electronic circuit elements, and output relevant information in both visual and text formats.

## Steps to Run the Project Locally

### **Step 1: Create a Project Folder**
Create a folder for the project, e.g., `server`.

```bash
mkdir server
cd server
```

### **Step 2: Set Up a Virtual Environment**
```bash
python -m venv .venv
```

This will create a .venv folder in your server directory that contains the virtual environment.

To activate the virtual environment use following command:
```bash
.venv\Scripts\activate
```


### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```
Ensure that the virtual environment is activated before running the above command.


### **Step 4: Run the Application**
```bash
python app.py
```
This will launch the server and run the Flask application locally.

### **Step 5: Access the Website**
Once the server is running, follow the link provided in the terminal (typically http://127.0.0.1:5000) to visit the website and interact with the application.

## **Folder Structure**
```bash
uploads/:
``` 
This folder will contain the images that are uploaded and intermediate processed images (e.g., black and white images or images with bounding boxes around detected elements).

```bash
outputs/:
``` 
This folder will contain text files, such as output.txt, which contain textual information about the circuit elements detected in the image and their connections to nodes.

### **Additional Information**
 Ensure that all dependencies are correctly installed and that the virtual environment is activated whenever running the project. If you encounter any issues, make sure to check the requirements.txt file for the correct package versions. 



