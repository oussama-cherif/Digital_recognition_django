from django.http import JsonResponse
from .utils import predict_digit_from_image
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
from django.core.files.storage import FileSystemStorage
from .utils import predict_digit_from_image, preprocess_image
import base64
import logging  # Add this line

logger = logging.getLogger(__name__)

@csrf_exempt
def predict_digit(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data['image_data']
        logger.debug(f"Received image data: {image_data[:30]}")  # Log part of the image data
        predicted_digit = predict_digit_from_image(image_data)
        logger.debug(f"Predicted digit: {predicted_digit}")  # Log the predicted digit
        return JsonResponse({'predicted_digit': int(predicted_digit)})
    else:
        return JsonResponse({'error': 'Send a POST request with proper data'})
    

def upload_and_predict_digit(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES.get('digit_image')
        if uploaded_file:
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_url = fs.url(filename)
            full_file_path = fs.path(filename)
            
            with open(full_file_path, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
                predicted_digit = predict_digit_from_image(base64_encoded_data)
            
            context['predicted_digit'] = predicted_digit
            context['uploaded_file_url'] = uploaded_file_url
    return render(request, 'digitrecognition/upload.html', context)

