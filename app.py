import warnings
import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import socket
import qrcode

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load multiple pre-trained models
try:
    model1 = load_model('Models/model (1).keras')  # MNIST
    model6 = load_model('Models/model1.keras')
    model7 = load_model('Models/model2.keras')
    model8 = load_model('Models/model_mnist.keras')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

def ensemble_predict(img):
    # Get predictions from all models
    preds = []
    confidences = []
    for model in [model1, model6, model7, model8]:
        pred = model.predict(img)[0]
        preds.append(np.argmax(pred))
        confidences.append(max(pred))
    
    # Voting: Majority vote with confidence tiebreaker if needed
    votes = np.bincount(preds, minlength=10)
    
    final_pred = np.argmax(votes)
    print(final_pred,"->",preds)
    
    # If there's a tie (multiple classes with max votes), use the highest confidence
    max_votes = np.max(votes)
    if np.sum(votes == max_votes) > 1:
        tie_indices = np.where(votes == max_votes)[0]
        final_pred = tie_indices[np.argmax([confidences[i] for i in range(3) if preds[i] in tie_indices])]
    
    # Calculate average confidence for the winning class
    confidence = np.mean([conf for pred, conf in zip(preds, confidences) if pred == final_pred]) * 100
    
    return final_pred, int(confidence)

def process_image(image_data):
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data URL prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        orig_width, orig_height = image.size
        # Calculate white background size (15% larger than the image)
        scale_factor = 2.75  # Use 1.15 for 15% larger, or 1.25 for 25% larger
        bg_width = int(image.width * scale_factor)
        bg_height = int(image.height * scale_factor)
        
        
        # Ensure even dimensions if needed
        if bg_width % 2 != 0:
            bg_width += 1
        if bg_height % 2 != 0:
            bg_height += 1
        
        # Create white background
        white_bg = Image.new("RGB", (bg_width, bg_height), (255, 255, 255))
        
        # Calculate position to center the image
        x_offset = (bg_width - image.width) // 2
        y_offset = (bg_height - image.height) // 2
        
        # Paste the original image onto the white background, handling transparency
        white_bg.paste(image, (x_offset, y_offset), mask=image if image.mode == "RGBA" else None)
        
        # Convert PIL image to OpenCV format
        image = np.array(white_bg)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu thresholding
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, "No digits detected"
        
        # Process all significant contours
        results = []
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        
        for cnt in contours:
            # Filter small contours (noise)
            area = cv2.contourArea(cnt)
            if area < 100:  # Adjust threshold as needed
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Extract ROI
            roi = th_up[y - top:y + h + 2*bottom+int(bottom*3/4), x - left:x + w + 2*right+int(right*3/4)]
            
            if roi.size == 0:
                continue
                
            # Resize and prepare for model
            img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            img = img.reshape(1, 28, 28, 1) / 255.0
            
            # Ensemble prediction
            final_pred, confidence = ensemble_predict(img)
            
            # Draw bounding box and prediction on the image
            digit_colors = {
                0: (255, 99, 132),   # red-pink
                1: (54, 162, 235),   # blue
                2: (255, 206, 86),   # yellow
                3: (75, 192, 192),   # teal
                4: (153, 102, 255),  # purple
                5: (255, 159, 64),   # orange
                6: (201, 203, 207),  # light gray
                7: (0, 204, 102),    # green
                8: (255, 102, 255),  # pink
                9: (100, 100, 100),  # dark gray
            }

            color = digit_colors.get(final_pred, (0, 0, 0))  # fallback to black
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{final_pred} ({confidence}%)", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Store result with bounding box
            results.append({
                "prediction": f"{final_pred} ({confidence}%)",
                "bbox": [x, y, w, h]
            })
        
        if not results:
            return None, None, "No valid digits detected"
            
        # Convert the image with bounding boxes back to base64
        left = x_offset
        top = y_offset
        right = x_offset + orig_width
        bottom = y_offset + orig_height
        image = image[y_offset:y_offset + orig_height, x_offset:x_offset + orig_width]
        _, buffer = cv2.imencode('.png', image)
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return results, image_base64, None
    except Exception as e:
        print(e)
        return None, None, f"Error processing image: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mobile_output')
def mobile_output():
    return render_template('mobile_output.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    
    results, image_base64, error = process_image(image_data)
    
    if error:
        return jsonify({'error': error}), 400
    
    return jsonify({'results': results, 'image': image_base64})

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return '127.0.0.1'

def generate_qr_code(host_ip, port):
    url = f"http://{host_ip}:{port}"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image.show()

if __name__ == '__main__':
    host_ip = get_local_ip()
    port = 5125
    generate_qr_code(host_ip, port)
    app.run(debug=True, host=host_ip, port=port)