
# Recognising Handwritten digits with deep learning

This is a web-based application for recognizing handwritten digits using multiple pre-trained Convolutional Neural Network (CNN) models with an ensemble prediction approach. Users can draw digits on a canvas, and the app processes the input to predict the digit with confidence scores, displaying results with bounding boxes around detected digits. The app is built with Flask for the backend, HTML/CSS/JavaScript for the frontend, and uses Keras models for digit recognition.



## Features

- **Interactive Canvas:** Draw digits using mouse or touch input.
- **Ensemble Prediction:** Combines predictions from four pre-trained CNN models for improved accuracy.
- **Responsive Design:** Optimized for both desktop and mobile devices, with a separate mobile output page for smaller screens.
- **Bounding Box Visualization:** Displays predicted digits with confidence scores and colored bounding boxes.
- **QR Code Access:** Generates a QR code for easy access to the app on local networks.
- **Real-time Processing:** Processes images with OpenCV and returns results instantly.

## Tech Stack

- **Backend:** Flask, Python, Keras, OpenCV, NumPy, PIL, qrcode
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Pre-trained CNN models for MNIST digit recognition
- **Dependencies:** See requirements.txt (not provided, but recommended to create)


## Prerequisites
- Python 3.8+
- pip for installing Python packages
- Git for cloning the repository
- A modern web browser (Chrome, Firefox, Safari, etc.)

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ZOROVIKKY/DigitRecognition.git
   cd DigitRecognition
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r Flask==2.3.2 opencv-python==4.8.0.76 numpy==1.24.3 tensorflow==2.13.0 Pillow==10.0.0 qrcode==7.4.2
   ```
6. **Run the Application**:
   ```bash
   python app.py
   ```
   - The app will start on `http://<local-ip>:5125`.
   - A QR code will be displayed in a pop-up window. Scan it with a mobile device to access the app on the same network.
## Usage

1. **Access the App**:
   - Open `http://<local-ip>:5125` in a browser or scan the QR code.
   - On desktop, the main page shows two canvases: one for drawing and one for output.
   - On mobile, the output is shown on a separate page after prediction.

2. **Draw a Digit**:
   - Use a mouse (desktop) or finger (mobile) to draw a digit on the canvas.
   - Click or tap the "Clear" button to reset the canvas.

3. **Predict**:
   - Click or-tap the "Predict" button to send the drawing to the backend.
   - The app processes the image, predicts the digit(s), and displays the result:
     - **Desktop**: The output canvas shows the processed image with bounding boxes and predictions.
     - **Mobile**: Redirects to a new page showing the output image.
   - Each predicted digit is highlighted with a colored bounding box and labeled with the digit and confidence score (e.g., "5 (95%)").

4. **Mobile Output**:
   - On mobile devices (screen width ≤ 768px), the predicted image is stored in `sessionStorage` and displayed on `/mobile_output`.
   - Click the "Back" button to return to the drawing page.
## How It Works

1. **Frontend**:
   - Users draw digits on a canvas using mouse or touch input.
   - The canvas is converted to a base64-encoded PNG image and sent to the `/predict` endpoint.
   - On desktop, the processed image is displayed on the output canvas.
   - On mobile, the image is stored in `sessionStorage` and shown on a separate page.

2. **Backend**:
   - Receives the base64 image, decodes it, and processes it using OpenCV (grayscale, thresholding, contour detection).
   - Extracts regions of interest (ROIs) for each detected digit.
   - Resizes ROIs to 28x28 pixels and feeds them into four pre-trained CNN models.
   - Uses ensemble voting (majority vote with confidence tiebreaker) to determine the final prediction.
   - Draws bounding boxes and predictions on the image with unique colors per digit.
   - Returns the processed image (base64) and prediction results to the frontend.

3. **Ensemble Prediction**:
   - Combines predictions from four models.
   - Uses majority voting; ties are resolved by selecting the prediction with the highest confidence.
   - Returns the predicted digit and average confidence score.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or feedback, please open an issue on GitHub.

