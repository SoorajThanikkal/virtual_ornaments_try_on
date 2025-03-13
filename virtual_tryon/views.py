import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import mediapipe as mp
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from .models import OrnamentProduct
import uuid

# Initialize MediaPipe components
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

# Initialize face mesh, face detection, and pose detection
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5)

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5)

def home(request):
    """Home page view"""
    return render(request, 'home.html')

def product_list(request):
    """View to display all products with filters"""
    ornament_type = request.GET.get('type', None)
    
    if ornament_type:
        products = OrnamentProduct.objects.filter(ornament_type=ornament_type)
    else:
        products = OrnamentProduct.objects.all()
        
    context = {
        'products': products,
        'ornament_types': OrnamentProduct.ORNAMENT_TYPES
    }
    return render(request, 'product_list.html', context)

def product_detail(request, product_id):
    """View to display product details and try-on option"""
    product = get_object_or_404(OrnamentProduct, id=product_id)
    context = {
        'product': product
    }
    return render(request, 'product_detail.html', context)

def try_on_page(request, product_id):
    """View for the try-on page"""
    product = get_object_or_404(OrnamentProduct, id=product_id)
    context = {
        'product': product
    }
    return render(request, 'try_on.html', context)

def process_try_on(request, product_id):
    """View to process the try-on operation"""
    if request.method != 'POST' or 'person_image' not in request.FILES:
        return JsonResponse({'error': 'No image uploaded'}, status=400)
    
    product = get_object_or_404(OrnamentProduct, id=product_id)
    
    # Save the uploaded person image temporarily
    person_image_file = request.FILES['person_image']
    person_image_path = os.path.join(settings.MEDIA_ROOT, 'temp', f'{uuid.uuid4()}.jpg')
    os.makedirs(os.path.dirname(person_image_path), exist_ok=True)
    
    with open(person_image_path, 'wb+') as destination:
        for chunk in person_image_file.chunks():
            destination.write(chunk)
    
    # Get ornament image path
    ornament_image_path = os.path.join(settings.MEDIA_ROOT, product.image.name)
    
    # Process images
    result_image, message = process_images(person_image_path, ornament_image_path, product.ornament_type)
    
    if result_image is not None:
        # Save the result image
        result_image_filename = f"result_{uuid.uuid4()}.jpg"
        result_image_path = os.path.join(settings.MEDIA_ROOT, 'results', result_image_filename)
        os.makedirs(os.path.dirname(result_image_path), exist_ok=True)
        cv2.imwrite(result_image_path, result_image)
        
        # Clean up the temporary person image
        if os.path.exists(person_image_path):
            os.remove(person_image_path)
            
        result_url = os.path.join(settings.MEDIA_URL, 'results', result_image_filename)
        return JsonResponse({'success': True, 'message': 'Try-on successful', 'result_image': result_url})
    else:
        # Clean up the temporary person image
        if os.path.exists(person_image_path):
            os.remove(person_image_path)
        return JsonResponse({'success': False, 'message': message})

# Function to remove background from ornament
def remove_background(image):
    # Convert to RGBA if not already
    if image.shape[2] == 3:
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        image_rgba = image

    # Simple background removal - convert white/light backgrounds to transparent
    # Create a mask for light pixels
    mask = cv2.inRange(image_rgba[:,:,:3], np.array([200, 200, 200]), np.array([255, 255, 255]))

    # Set alpha channel to 0 where mask is white (background)
    image_rgba[:, :, 3] = np.where(mask == 255, 0, 255)

    return image_rgba

# Function to detect facial landmarks using MediaPipe
def detect_face_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    return results

# Function to detect face using MediaPipe Face Detection
def detect_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    return results

# Function to detect body landmarks using MediaPipe
def detect_body_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results

# Function to get face size for better scaling
def get_face_size(image):
    # Try to get face size using face detection first
    face_results = detect_face(image)
    if face_results.detections:
        detection = face_results.detections[0]
        bounding_box = detection.location_data.relative_bounding_box

        # Convert relative coordinates to absolute
        height, width, _ = image.shape
        x_min = int(bounding_box.xmin * width)
        y_min = int(bounding_box.ymin * height)
        box_width = int(bounding_box.width * width)
        box_height = int(bounding_box.height * height)

        return box_width, box_height

    # Fallback to face mesh if face detection fails
    face_mesh_results = detect_face_landmarks(image)
    if face_mesh_results.multi_face_landmarks:
        landmarks = face_mesh_results.multi_face_landmarks[0].landmark

        # Get face bounds from landmarks
        min_x = min(landmarks, key=lambda l: l.x).x
        max_x = max(landmarks, key=lambda l: l.x).x
        min_y = min(landmarks, key=lambda l: l.y).y
        max_y = max(landmarks, key=lambda l: l.y).y

        height, width, _ = image.shape
        face_width = int((max_x - min_x) * width)
        face_height = int((max_y - min_y) * height)

        return face_width, face_height

    # If all detection methods fail, return a default proportion of image size
    height, width, _ = image.shape
    return width // 3, height // 3  # Default to 1/3 of image dimensions

# Function to place glasses on face with adaptive sizing
def place_glasses(person_image, glasses_image):
    # Get face landmarks
    face_results = detect_face_landmarks(person_image)

    if not face_results.multi_face_landmarks:
        return person_image, False

    face_landmarks = face_results.multi_face_landmarks[0]

    # Get key landmarks for glasses placement
    left_eye = np.array([int(face_landmarks.landmark[33].x * person_image.shape[1]),
                         int(face_landmarks.landmark[33].y * person_image.shape[0])])
    right_eye = np.array([int(face_landmarks.landmark[263].x * person_image.shape[1]),
                          int(face_landmarks.landmark[263].y * person_image.shape[0])])

    # Calculate distance between eyes for horizontal scaling
    eye_distance = np.linalg.norm(right_eye - left_eye)

    # Get face width for better proportional sizing
    face_width, face_height = get_face_size(person_image)

    # Calculate temple width (usually wider than eye distance)
    temple_width = face_width * 0.9  # 90% of face width

    # Adjust glasses width based on temple width
    glasses_width = int(temple_width)
    glasses_height = int(glasses_width * glasses_image.shape[0] / glasses_image.shape[1])

    # Fine-tune vertical position based on face proportions
    vertical_offset_factor = face_height * 0.05  # 5% of face height

    # Resize glasses
    glasses_resized = cv2.resize(glasses_image, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

    # Calculate position to place glasses
    center_x = (left_eye[0] + right_eye[0]) // 2
    center_y = (left_eye[1] + right_eye[1]) // 2

    # Adjust vertical position (slightly above eyes)
    center_y -= int(vertical_offset_factor)

    # Create a copy of the person image
    result_image = person_image.copy()

    # Define the region to place glasses
    x_offset = center_x - glasses_width // 2
    y_offset = center_y - glasses_height // 2

    # Place glasses on the face
    for i in range(glasses_height):
        for j in range(glasses_width):
            if (y_offset + i >= 0 and y_offset + i < result_image.shape[0] and
                x_offset + j >= 0 and x_offset + j < result_image.shape[1]):

                # For 4-channel images (with alpha)
                if glasses_resized.shape[2] == 4 and glasses_resized[i, j, 3] > 0:
                    alpha = glasses_resized[i, j, 3] / 255.0
                    result_image[y_offset + i, x_offset + j] = (
                        (1 - alpha) * result_image[y_offset + i, x_offset + j] +
                        alpha * glasses_resized[i, j, :3]
                    ).astype(np.uint8)

    return result_image, True

# Function to place necklace on person with adaptive sizing
def place_necklace(person_image, necklace_image):
    # Get body landmarks
    body_results = detect_body_landmarks(person_image)

    if not body_results.pose_landmarks:
        return person_image, False

    landmarks = body_results.pose_landmarks

    # Get shoulder landmarks
    left_shoulder = np.array([int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * person_image.shape[1]),
                              int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * person_image.shape[0])])
    right_shoulder = np.array([int(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * person_image.shape[1]),
                               int(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * person_image.shape[0])])

    # Get face landmarks to calculate neck width
    face_results = detect_face_landmarks(person_image)
    neck_width = None

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]

        # Using jaw landmarks (jawline endpoints) to estimate neck width
        left_jawline = np.array([int(face_landmarks.landmark[149].x * person_image.shape[1]),
                               int(face_landmarks.landmark[149].y * person_image.shape[0])])
        right_jawline = np.array([int(face_landmarks.landmark[378].x * person_image.shape[1]),
                                int(face_landmarks.landmark[378].y * person_image.shape[0])])

        # The neck is typically narrower than the jaw
        jaw_width = np.linalg.norm(right_jawline - left_jawline)
        neck_width = jaw_width * 0.8  # Neck is approximately 80% of jaw width
    else:
        # Fallback if no face landmarks
        shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder)
        neck_width = shoulder_distance * 0.4  # Neck is typically 40% of shoulder width

    # Calculate position for necklace
    shoulder_distance = np.linalg.norm(right_shoulder - left_shoulder)
    center_x = (left_shoulder[0] + right_shoulder[0]) // 2

    # Estimate neck position
    if hasattr(mp_pose.PoseLandmark, 'NOSE'):
        nose_y = int(landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * person_image.shape[0])
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2

        # Neck is positioned between nose and shoulders, closer to shoulders
        neck_y = nose_y + int((shoulder_y - nose_y) * 0.3)
    else:
        # Fallback if nose isn't detected
        neck_y = (left_shoulder[1] + right_shoulder[1]) // 2 - int(shoulder_distance * 0.2)

    # Use neck width for necklace sizing
    necklace_width = int(neck_width * 4.2)  # Necklace typically drapes beyond neck width

    # Maintain aspect ratio
    necklace_height = int(necklace_width * necklace_image.shape[0] / necklace_image.shape[1])

    # Resize necklace
    necklace_resized = cv2.resize(necklace_image, (necklace_width, necklace_height), interpolation=cv2.INTER_AREA)

    # Create a copy of the person image
    result_image = person_image.copy()

    # Define the region to place necklace
    x_offset = center_x - necklace_width // 2
    y_offset = neck_y - int(necklace_height * 0.001)   # Position at the neck

    # Place necklace on the person
    for i in range(necklace_height):
        for j in range(necklace_width):
            if (y_offset + i >= 0 and y_offset + i < result_image.shape[0] and
                x_offset + j >= 0 and x_offset + j < result_image.shape[1]):

                # For 4-channel images (with alpha)
                if necklace_resized.shape[2] == 4 and necklace_resized[i, j, 3] > 0:
                    alpha = necklace_resized[i, j, 3] / 255.0
                    result_image[y_offset + i, x_offset + j] = (
                        (1 - alpha) * result_image[y_offset + i, x_offset + j] +
                        alpha * necklace_resized[i, j, :3]
                    ).astype(np.uint8)

    return result_image, True

# Function to place earrings with adaptive sizing
def place_earrings(person_image, earring_image):
    # Get face landmarks
    face_results = detect_face_landmarks(person_image)

    if not face_results.multi_face_landmarks:
        return person_image, False

    face_landmarks = face_results.multi_face_landmarks[0]

    # Get ear landmarks (234 is right ear, 454 is left ear in MediaPipe Face Mesh)
    left_ear = np.array([int(face_landmarks.landmark[234].x * person_image.shape[1]),
                         int(face_landmarks.landmark[234].y * person_image.shape[0])])
    right_ear = np.array([int(face_landmarks.landmark[454].x * person_image.shape[1]),
                          int(face_landmarks.landmark[454].y * person_image.shape[0])])

    # Get face size for proportional scaling
    face_width, face_height = get_face_size(person_image)

    # Calculate appropriate earring size based on face proportions
    face_ratio = face_width / face_height

    # Adjust earring size based on face shape
    if face_ratio < 0.7:  # Narrower face
        earring_scale = 0.15  # Smaller earrings
    elif face_ratio > 0.9:  # Wider face
        earring_scale = 0.25  # Larger earrings
    else:
        earring_scale = 0.2  # Default

    # Calculate earring size proportional to face dimensions
    earring_size = int(face_width * earring_scale)
    earring_resized = cv2.resize(earring_image, (earring_size, earring_size), interpolation=cv2.INTER_AREA)

    # Create a copy of the person image
    result_image = person_image.copy()

    # Place earrings on both ears
    for ear_pos in [left_ear, right_ear]:
        # Create a little gap from the ear by adjusting x_offset
        if np.array_equal(ear_pos, left_ear):
            x_offset = ear_pos[0] - earring_size // 2 - int(earring_size * 0.15)
        else:  # right ear
            x_offset = ear_pos[0] - earring_size // 2 + int(earring_size * 0.15)

        # Move earrings downward by adding pixels to y_offset
        y_offset = ear_pos[1] + int(earring_size * 0.2)

        # Place earring
        for i in range(earring_size):
            for j in range(earring_size):
                if (y_offset + i >= 0 and y_offset + i < result_image.shape[0] and
                    x_offset + j >= 0 and x_offset + j < result_image.shape[1]):

                    # For 4-channel images (with alpha)
                    if earring_resized.shape[2] == 4 and earring_resized[i, j, 3] > 0:
                        alpha = earring_resized[i, j, 3] / 255.0
                        result_image[y_offset + i, x_offset + j] = (
                            (1 - alpha) * result_image[y_offset + i, x_offset + j] +
                            alpha * earring_resized[i, j, :3]
                        ).astype(np.uint8)

    return result_image, True

# Main function to process images
def process_images(person_image_path, ornament_image_path, ornament_type):
    # Read images
    person_image = cv2.imread(person_image_path)
    ornament_image = cv2.imread(ornament_image_path, cv2.IMREAD_UNCHANGED)

    # Check if images were loaded properly
    if person_image is None or ornament_image is None:
        return None, "Failed to load images"

    # Preprocess the ornament image (remove background if needed)
    if ornament_image.shape[2] == 3:  # If RGB (no alpha channel)
        ornament_image = remove_background(ornament_image)

    # Process based on ornament type
    success = False
    if ornament_type.lower() == 'glasses':
        result_image, success = place_glasses(person_image, ornament_image)
    elif ornament_type.lower() == 'necklace':
        result_image, success = place_necklace(person_image, ornament_image)
    elif ornament_type.lower() == 'earrings':
        result_image, success = place_earrings(person_image, ornament_image)
    else:
        return None, f"Unsupported ornament type: {ornament_type}"

    if not success:
        return None, f"Failed to place {ornament_type}. Could not detect required landmarks."

    return result_image, "Success"