from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

# Load the pre-trained model
model_path = 'digitrecognition/digit_rec_model/best_model.keras'
model = load_model(model_path)


def preprocess_image(image_data):
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    img = Image.open(image_bytes).convert('L')

    # Resize image, ensure it matches the model's expected input
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # Ensure this shape matches the model input

    return img_array

def predict_digit_from_image(image_data):
    # Preprocess the image
    img = preprocess_image(image_data)

    # Predict the digit
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions, axis=1)[0]
    return predicted_digit