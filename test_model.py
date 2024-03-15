from digitrecognition.utils import preprocess_image, model  # Update with correct import paths
import numpy as np
import base64

test_image_path = 'digitrecognition/digit_rec_model/28-3-28-image-of-a-handwritten-7-divided-into-four-14-3-14-images.png'

with open(test_image_path, "rb") as image_file:
    base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    processed_image = preprocess_image(base64_encoded_data)
    prediction = model.predict(processed_image)
    print(np.argmax(prediction, axis=1)[0])  # This should print the correct digit
