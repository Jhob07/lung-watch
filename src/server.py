from flask import Flask, request, jsonify
from flask_cors import CORS
from database import Database
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from HybridModel import HybridModel
import base64
import io
from datetime import datetime
import logging

# ✅ Logging setup
logging.basicConfig(filename='server.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Ensure segmentations directory exists
os.makedirs('uploads', exist_ok=True)
os.makedirs('src/static/segmentations', exist_ok=True)

# ✅ Config
THRESHOLD = 0.45

# ✅ Init Flask
app = Flask(__name__)
CORS(app)

# ✅ Init database
db = Database()

# ✅ Load model
try:
    print("🚀 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    model = HybridModel().to(device)
    model_path = 'src/Models/model_final.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Model loaded and set to eval mode")
except Exception as e:
    print(f"❌ Model load error: {e}")
    logging.error(f"Model load error: {e}")
    raise

# ✅ Image processing

def analyze_xray(image_data):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((192, 192))

        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            seg_mask, cls_output = model(image_tensor)
            probs = torch.sigmoid(cls_output).item()
            predicted = 1 if probs > THRESHOLD else 0
            confidence = probs if predicted == 1 else 1 - probs

        result = "Atelectasis" if predicted == 1 else "Normal"
        confidence_str = f"{confidence:.2%}"

        # ✅ Save segmentation
        seg = seg_mask.squeeze().cpu().numpy()
        seg = (seg - seg.min()) / (seg.max() - seg.min())
        seg = (seg * 255).astype(np.uint8)
        seg_image = Image.fromarray(seg)
        seg_filename = f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        seg_path = os.path.join("src/static/segmentations", seg_filename)
        seg_image.save(seg_path)

        logging.info(f"Prediction: {result} | Confidence: {confidence_str}")

        return {
            "result": result,
            "confidence": confidence_str,
            "segmentation_path": os.path.join("static", "segmentations", seg_filename)
        }

    except Exception as e:
        logging.error(f"Analysis error: {e}")
        print(f"❌ Analysis error: {e}")
        return {"error": str(e)}

# ✅ API routes
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    result = analyze_xray(data['image'])
    if "error" in result:
        return jsonify(result), 500
    return jsonify({"success": True, **result}), 200

@app.route('/history', methods=['GET'])
def get_history():
    # Placeholder - implement actual DB retrieval here
    return jsonify({'success': True, 'history': []}), 200

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    required = ['license_number', 'name', 'password']
    if not all(field in data for field in required):
        return jsonify({'success': False, 'message': 'Missing fields'}), 400

    success, result = db.register_user(
        data['license_number'], data['name'], data['password'],
        data.get('email'), data.get('specialization'), data.get('hospital')
    )
    if success:
        return jsonify({'success': True, 'user': result}), 201
    return jsonify({'success': False, 'message': result}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'license_number' not in data or 'password' not in data:
        return jsonify({'success': False, 'message': 'Missing credentials'}), 400

    success, result = db.login_user(data['license_number'], data['password'])
    if success:
        return jsonify({'success': True, 'user': result}), 200
    return jsonify({'success': False, 'message': result}), 401

if __name__ == '__main__':
    app.run(debug=True, port=5000)