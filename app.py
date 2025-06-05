import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import cv2

# CLASSES

class AdaptiveHistEqualization:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, image):
        image_np = np.array(image)  # Convert PIL Image to NumPy array
        if len(image_np.shape) == 2:  # Grayscale
            image_np = self.clahe.apply(image_np)
        else:  # Color
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            image_np[:, :, 0] = self.clahe.apply(image_np[:, :, 0])
            image_np = cv2.cvtColor(image_np, cv2.COLOR_LAB2RGB)
        return image_np

##################################################################################################################

app = Flask(__name__)

# Configure upload folder for security and organization
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create the folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Dummy Model Prediction Function ---
# In a real application, you would load your actual trained model here
# (e.g., using TensorFlow, PyTorch, scikit-learn).
# This function simulates model output for demonstration purposes.
def get_model_prediction(image_path):
    """
    Simulates a model prediction.
    In a real scenario, you'd load the image, preprocess it,
    and then pass it through your actual model.
    """
    try:
        img = Image.open(image_path).convert('RGB')

        multiple_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            AdaptiveHistEqualization(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Processing image: {image_path}")


        img = multiple_transforms(img)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

        loaded_resNet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for name, param in loaded_resNet.named_parameters():
            if not (name.startswith('layer4') or name.startswith('fc')):
                param.requires_grad = False

        input_layer = loaded_resNet.fc.in_features
        num_classes = 3
        loaded_resNet.fc = nn.Sequential(
            nn.Linear(input_layer, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, num_classes),
        )

        model_save_path = 'my_skin_disease_model.pth'

        loaded_resNet.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))

        loaded_resNet.eval()

        print("Model loaded successfully!")

        img = img.unsqueeze(0)

        with torch.no_grad():
            output = loaded_resNet(img)

        probabilities = F.softmax(output, dim=1)
        probabilities = probabilities.squeeze()

        # Convert to percentages
        percentages = (probabilities * 100).tolist()
        
        # Example class names (replace with your actual class names)
        class_names = ["acne", "eczema", "vitiligo"]
        
        results = []
        for i, prob in enumerate(percentages):
            results.append({"class": class_names[i], "percentage": f"{prob:.2f}%"})
            
        return results

    except Exception as e:
        print(f"Error during dummy model prediction: {e}")
        return None

# --- Routes ---

# Serves the main HTML page
@app.route('/')
def index():
    return render_template('index.html') # Ensure your HTML is in a 'templates' folder or correctly referenced

# Handles image uploads and returns predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction from your model
        predictions = get_model_prediction(filepath)
        
        if predictions:
            # os.remove(filepath)
            return jsonify({'predictions': predictions})
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # For development, run in debug mode. In production, use a production WSGI server.
    app.run(debug=True, port=5000)