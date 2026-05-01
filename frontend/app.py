from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector, os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

#conda run --no-capture-output -n brain_stroke python app.py


import os

app = Flask(__name__)

mydb = mysql.connector.connect(
    host     = os.environ.get("DB_HOST",     "localhost"),
    user     = os.environ.get("DB_USER",     "root"),
    password = os.environ.get("DB_PASSWORD", ""),
    port     = int(os.environ.get("DB_PORT", "3306")),
    database = os.environ.get("DB_NAME",     "stroke")
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/view_data', methods=["GET", "POST"])
def view_data():
    if request.method == "POST":
        n = request.form['n']

        excel_file = "#"
        df = pd.read_excel(excel_file)
        df = df.head(n)
        df = df.to_html()

        return render_template('view_data.html', data = df)
    return render_template('view_data.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join(r'static\saved_images', fn)
        myfile.save(mypath)

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image transformations
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define the model class (same as the one used during training)
        class MobileNetModel(nn.Module):
            def __init__(self, num_classes):
                super(MobileNetModel, self).__init__()
                self.mobilenet = models.mobilenet_v2(pretrained=True)
                num_features = self.mobilenet.classifier[1].in_features
                self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

            def forward(self, x):
                return self.mobilenet(x)

        # Load the trained model
        model = MobileNetModel(num_classes=2)
        model.load_state_dict(torch.load("mobilenet.pt", map_location=torch.device('cpu')))
        model = model.to(device)
        model.eval()

        # Load and preprocess the image
        original_image = Image.open(mypath).convert('RGB')
        image_tensor = image_transform(original_image).unsqueeze(0).to(device)

        # ── Grad-CAM: register hooks on MobileNetV2's last conv layer ──
        activations = []
        gradients  = []

        def _forward_hook(module, inp, out):
            activations.append(out)

        def _backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = model.mobilenet.features[-1]
        fwd_handle = target_layer.register_forward_hook(_forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(_backward_hook)

        # ── Classification threshold ─────────────────────────────────────────
        # Stroke is predicted only when P(Stroke) >= THRESHOLD.
        # Lowering the threshold makes the model more sensitive (catches more strokes).
        # Raising it makes it more specific (fewer false alarms).
        THRESHOLD = 0.5   # ← change this value to tune sensitivity (range 0.0–1.0)

        # Forward pass (gradients enabled — no torch.no_grad())
        output = model(image_tensor)

        # Convert raw logits → probabilities via softmax
        probabilities  = torch.softmax(output, dim=1)
        normal_prob    = probabilities[0, 0].item()   # P(Normal)
        stroke_prob    = probabilities[0, 1].item()   # P(Stroke)

        # Apply threshold to decide the class
        predicted_class = 1 if stroke_prob >= THRESHOLD else 0

        # Backward pass for the predicted class
        model.zero_grad()
        output[0, predicted_class].backward()

        # Remove hooks
        fwd_handle.remove()
        bwd_handle.remove()

        # ── Compute Grad-CAM heatmap ──
        activation = activations[0].detach()   # [1, C, H, W]
        gradient   = gradients[0].detach()     # [1, C, H, W]

        # Global-average-pool gradients → channel weights
        weights = gradient.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * activation).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize cam to 224×224 with PIL (avoids OpenCV dependency)
        cam_pil = Image.fromarray(np.uint8(255 * cam))
        cam_pil = cam_pil.resize((224, 224), Image.LANCZOS)
        cam_np  = np.array(cam_pil) / 255.0

        # Apply 'jet' colormap via matplotlib
        cmap         = plt.get_cmap('jet')
        heatmap_rgba = cmap(cam_np)
        heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

        # Blend heatmap with original (50 / 50)
        original_resized = original_image.resize((224, 224))
        original_np      = np.array(original_resized)
        blended          = (0.5 * original_np + 0.5 * heatmap_rgb).astype(np.uint8)

        # Save Grad-CAM image next to the original
        gradcam_filename = 'gradcam_' + fn
        gradcam_path = os.path.join(r'static\saved_images', gradcam_filename)
        Image.fromarray(blended).save(gradcam_path)

        # Map prediction to human-readable label
        label_mapping   = {0: "Normal", 1: "Stroke"}
        predicted_label = label_mapping.get(predicted_class, "Unknown")

        print(f"Stroke probability : {stroke_prob:.4f}  |  Threshold: {THRESHOLD}  |  Result: {predicted_label}")

        return render_template(
            'prediction.html',
            prediction   = predicted_label,
            path         = mypath,
            gradcam_path = gradcam_path,
            stroke_prob  = round(stroke_prob  * 100, 1),   # as percentage
            normal_prob  = round(normal_prob  * 100, 1),
            threshold    = round(THRESHOLD    * 100, 1),
        )
    return render_template('prediction.html')


@app.route('/graph')
def graph():
    return render_template('graph.html')


if __name__ == '__main__':
    app.run(debug = True)