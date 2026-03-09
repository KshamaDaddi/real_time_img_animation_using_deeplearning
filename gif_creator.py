import io
import base64
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2

def create_animated_gif(image_data, animation_type='face-move', speed='medium'):
    """
    Create an animated GIF from a base64 encoded image with various animation effects.
    
    Args:
        image_data (str): Base64 encoded image data
        animation_type (str): Type of animation ('face-move', 'blink', 'wave', 'bounce', 'cry', 'talk', 'wink')
        speed (str): Animation speed ('slow', 'medium', 'fast')
    
    Returns:
        str: Base64 encoded GIF data
    """
    try:
        # Decode the base64 image
        if image_data.startswith('data:image'):
            header, base64_data = image_data.split(',', 1)
        else:
            base64_data = image_data
            
        image_bytes = base64.b64decode(base64_data)
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGBA if needed
        if original_image.mode != 'RGBA':
            original_image = original_image.convert('RGBA')
            
        # Create frames for the animation
        frames = []
        
        # Determine number of frames and delay based on speed
        if speed == 'slow':
            num_frames = 12
            duration = 200
        elif speed == 'fast':
            num_frames = 8
            duration = 100
        else:  # medium
            num_frames = 10
            duration = 150
            
        # Create animation based on type
        if animation_type == 'face-move':
            frames = _create_face_movement_animation(original_image, num_frames)
        elif animation_type == 'blink':
            frames = _create_blinking_animation(original_image, num_frames)
        elif animation_type == 'wave':
            frames = _create_wave_animation(original_image, num_frames)
        elif animation_type == 'bounce':
            frames = _create_bounce_animation(original_image, num_frames)
        elif animation_type == 'cry':
            frames = _create_cry_animation(original_image, num_frames)
        elif animation_type == 'talk':
            frames = _create_talking_animation(original_image, num_frames)
        elif animation_type == 'wink':
            frames = _create_winking_animation(original_image, num_frames)
        else:
            # Default to face movement
            frames = _create_face_movement_animation(original_image, num_frames)
            
        # Create GIF from frames
        if frames:
            # Save to bytes
            gif_buffer = io.BytesIO()
            frames[0].save(
                gif_buffer,
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0  # Infinite loop
            )
            gif_buffer.seek(0)
            
            # Encode to base64
            gif_base64 = base64.b64encode(gif_buffer.getvalue()).decode('utf-8')
            return f"data:image/gif;base64,{gif_base64}"
        else:
            # If animation creation failed, return original image as PNG
            png_buffer = io.BytesIO()
            original_image.save(png_buffer, format='PNG')
            png_buffer.seek(0)
            png_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{png_base64}"
            
    except Exception as e:
        print(f"Error creating animated GIF: {str(e)}")
        # Return original image if GIF creation fails
        return image_data

def _detect_face_and_features_opencv(image):
    """
    Detect face and facial features in the image using OpenCV.
    
    Returns:
        dict: Dictionary containing face coordinates and features or None if no face detected
    """
    try:
        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Convert RGB to grayscale for face detection
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load eye cascade classifier
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Get the first face (assuming single face for simplicity)
        (x, y, w, h) = faces[0]
        
        # Calculate face center and other key points
        face_center = (x + w // 2, y + h // 2)
        
        # Detect eyes within the face region
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10)
        )
        
        left_eye = None
        right_eye = None
        
        # Sort eyes by x-coordinate to identify left and right eyes
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])  # Sort by x-coordinate
            left_eye_coords = eyes_sorted[0]
            right_eye_coords = eyes_sorted[1]
            
            # Calculate actual positions
            left_eye = (x + left_eye_coords[0] + left_eye_coords[2] // 2, 
                        y + left_eye_coords[1] + left_eye_coords[3] // 2)
            right_eye = (x + right_eye_coords[0] + right_eye_coords[2] // 2, 
                         y + right_eye_coords[1] + right_eye_coords[3] // 2)
        else:
            # Fallback to estimated positions
            eye_y = y + h // 3
            left_eye = (x + w // 4, eye_y)
            right_eye = (x + 3 * w // 4, eye_y)
        
        # Estimate mouth position (roughly 2/3 from top)
        mouth = (face_center[0], y + 2 * h // 3)
        
        return {
            'face_center': face_center,
            'face_bbox': (x, y, w, h),
            'left_eye': left_eye,
            'right_eye': right_eye,
            'mouth': mouth,
            'face_width': w,
            'face_height': h
        }
    except Exception as e:
        print(f"Error detecting face with OpenCV: {str(e)}")
        return None

def _create_face_movement_animation(image, num_frames):
    """Create a face movement animation by shifting the image horizontally."""
    frames = []
    width, height = image.size
    
    # Calculate maximum shift (5% of width)
    max_shift = int(width * 0.05)
    
    for i in range(num_frames):
        # Calculate shift amount (oscillating between -max_shift and max_shift)
        shift = int(max_shift * np.sin(2 * np.pi * i / num_frames))
        
        # Create new frame
        frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Paste the original image with horizontal shift
        frame.paste(image, (shift, 0))
        
        frames.append(frame)
        
    return frames

def _create_blinking_animation(image, num_frames):
    """Create a blinking animation by detecting eyes and animating them."""
    frames = []
    
    # Try to detect face and features
    face_data = _detect_face_and_features_opencv(image)
    
    width, height = image.size
    
    # Fallback positions if face detection fails
    if face_data:
        left_eye_center = face_data['left_eye']
        right_eye_center = face_data['right_eye']
        eye_width = face_data['face_width'] // 8
        eye_height = face_data['face_height'] // 12
    else:
        # Estimate eye positions
        face_center_x = width // 2
        face_center_y = height // 3
        left_eye_center = (face_center_x - width // 8, face_center_y)
        right_eye_center = (face_center_x + width // 8, face_center_y)
        eye_width = width // 16
        eye_height = height // 24
    
    for i in range(num_frames):
        # Create a copy of the image
        frame = image.copy()
        draw = ImageDraw.Draw(frame)
        
        # Calculate blink progress (close eyes for part of the animation)
        blink_progress = abs(np.sin(2 * np.pi * i / (num_frames / 2)))
        
        # Only blink for part of the cycle
        if blink_progress > 0.7:
            # Draw closed eyes (lines)
            blink_intensity = min(1.0, (blink_progress - 0.7) * 3.3)  # Scale to 0-1
            
            # Left eye
            left_line_y = left_eye_center[1] + int(eye_height * blink_intensity * 0.2)
            draw.line([
                (left_eye_center[0] - eye_width, left_line_y),
                (left_eye_center[0] + eye_width, left_line_y)
            ], fill=(255, 255, 255, 200), width=2)
            
            # Right eye
            right_line_y = right_eye_center[1] + int(eye_height * blink_intensity * 0.2)
            draw.line([
                (right_eye_center[0] - eye_width, right_line_y),
                (right_eye_center[0] + eye_width, right_line_y)
            ], fill=(255, 255, 255, 200), width=2)
        
        frames.append(frame)
        
    return frames

def _create_wave_animation(image, num_frames):
    """Create a wave animation by applying a sine wave distortion."""
    frames = []
    width, height = image.size
    
    for i in range(num_frames):
        # Calculate wave parameters
        amplitude = int(height * 0.02)  # 2% of height
        frequency = 2 * np.pi / width
        phase = 2 * np.pi * i / num_frames
        
        # Create displacement map
        displacement = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                displacement[y, x] = amplitude * np.sin(frequency * x + phase)
                
        # Apply displacement to image
        frame = _apply_displacement(image, displacement)
        frames.append(frame)
        
    return frames

def _create_bounce_animation(image, num_frames):
    """Create a bouncing animation by moving the image vertically."""
    frames = []
    width, height = image.size
    
    # Calculate maximum bounce (3% of height)
    max_bounce = int(height * 0.03)
    
    for i in range(num_frames):
        # Calculate vertical shift (oscillating between -max_bounce and max_bounce)
        shift = int(max_bounce * np.sin(2 * np.pi * i / num_frames))
        
        # Create new frame
        frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Paste the original image with vertical shift
        frame.paste(image, (0, shift))
        
        frames.append(frame)
        
    return frames

def _create_cry_animation(image, num_frames):
    """Create a crying animation with tears at detected eye positions."""
    frames = []
    
    # Try to detect face and features
    face_data = _detect_face_and_features_opencv(image)
    
    width, height = image.size
    
    # Fallback positions if face detection fails
    if face_data:
        left_eye = face_data['left_eye']
        right_eye = face_data['right_eye']
        eye_y = left_eye[1]
    else:
        # Estimate eye positions
        eye_y = height // 3
        left_eye = (width // 2 - width // 8, eye_y)
        right_eye = (width // 2 + width // 8, eye_y)
    
    # Tear properties
    tear_radius = max(2, int(width * 0.01))
    max_tear_length = int(height * 0.2)
    
    for i in range(num_frames):
        # Create a copy of the image
        frame = image.copy()
        draw = ImageDraw.Draw(frame)
        
        # Calculate cry intensity (oscillating between 0.2 and 1)
        cry_intensity = 0.2 + 0.8 * abs(np.sin(2 * np.pi * i / num_frames))
        
        # Number of tears based on intensity
        num_tears = max(1, int(5 * cry_intensity))
        
        # Draw tears falling from eyes
        for j in range(num_tears):
            # Animate tear position
            tear_progress = (i + j * 0.5) % num_frames / num_frames
            tear_y_offset = int(tear_progress * max_tear_length)
            
            # Left eye tears
            tear_x = left_eye[0] + np.random.randint(-5, 5)
            tear_y_start = left_eye[1] + 10
            tear_y_end = tear_y_start + tear_y_offset
            
            if tear_y_end <= height:
                draw.ellipse([tear_x - tear_radius, tear_y_end - tear_radius, 
                              tear_x + tear_radius, tear_y_end + tear_radius], 
                             fill=(173, 216, 230, 180))  # Light blue tears
            
            # Right eye tears
            tear_x = right_eye[0] + np.random.randint(-5, 5)
            tear_y_start = right_eye[1] + 10
            tear_y_end = tear_y_start + tear_y_offset
            
            if tear_y_end <= height:
                draw.ellipse([tear_x - tear_radius, tear_y_end - tear_radius, 
                              tear_x + tear_radius, tear_y_end + tear_radius], 
                             fill=(173, 216, 230, 180))  # Light blue tears
        
        frames.append(frame)
        
    return frames

def _create_talking_animation(image, num_frames):
    """Create a talking animation by detecting mouth and animating it."""
    frames = []
    
    # Try to detect face and features
    face_data = _detect_face_and_features_opencv(image)
    
    width, height = image.size
    
    # Fallback positions if face detection fails
    if face_data:
        mouth_center = face_data['mouth']
        mouth_width = face_data['face_width'] // 2
        mouth_height = face_data['face_height'] // 10
    else:
        # Estimate mouth position
        face_center_x = width // 2
        face_center_y = height // 2
        mouth_center = (face_center_x, face_center_y + height // 8)
        mouth_width = width // 3
        mouth_height = height // 20
    
    mouth_area = (
        mouth_center[0] - mouth_width // 2,
        mouth_center[1] - mouth_height // 2,
        mouth_center[0] + mouth_width // 2,
        mouth_center[1] + mouth_height // 2
    )
    
    for i in range(num_frames):
        # Create a copy of the image
        frame = image.copy()
        
        # Calculate mouth movement (oscillating between open and closed)
        mouth_openness = abs(np.sin(2 * np.pi * i / (num_frames / 3)))  # Faster oscillation for natural speech
        
        # Draw a talking mouth
        draw = ImageDraw.Draw(frame)
        
        # Clear the mouth area
        draw.rectangle(mouth_area, fill=(0, 0, 0, 0))
        
        # Draw different mouth shapes based on openness
        if mouth_openness > 0.7:
            # Wide open mouth (O shape)
            draw.ellipse([
                mouth_area[0],
                mouth_center[1] - int(mouth_height * mouth_openness * 0.7),
                mouth_area[2],
                mouth_center[1] + int(mouth_height * mouth_openness * 0.7)
            ], outline=(255, 255, 255, 220), fill=(50, 50, 50, 180), width=3)
        elif mouth_openness > 0.3:
            # Medium open mouth (U shape)
            draw.rectangle([
                mouth_area[0],
                mouth_center[1] - int(mouth_height * mouth_openness * 0.4),
                mouth_area[2],
                mouth_center[1] + int(mouth_height * mouth_openness * 0.4)
            ], outline=(255, 255, 255, 200), fill=(50, 50, 50, 150), width=2)
        else:
            # Nearly closed mouth (line)
            draw.line([(mouth_area[0], mouth_center[1]), (mouth_area[2], mouth_center[1])], 
                      fill=(255, 255, 255, 200), width=2)
        
        frames.append(frame)
        
    return frames

def _create_winking_animation(image, num_frames):
    """Create a winking animation by detecting eyes and closing one."""
    frames = []
    
    # Try to detect face and features
    face_data = _detect_face_and_features_opencv(image)
    
    width, height = image.size
    
    # Fallback positions if face detection fails
    if face_data:
        right_eye_center = face_data['right_eye']
        eye_width = face_data['face_width'] // 8
        eye_height = face_data['face_height'] // 12
    else:
        # Estimate eye positions
        face_center_x = width // 2
        face_center_y = height // 3
        right_eye_center = (face_center_x + width // 8, face_center_y)
        eye_width = width // 16
        eye_height = height // 24
    
    for i in range(num_frames):
        # Create a copy of the image
        frame = image.copy()
        draw = ImageDraw.Draw(frame)
        
        # Calculate wink progress (close right eye for 40% of animation)
        wink_progress = abs(np.sin(2 * np.pi * i / num_frames))
        
        # Draw winking effect on right eye
        if wink_progress > 0.6:  # Only wink for part of the cycle
            # Draw closed right eye (line)
            wink_intensity = min(1.0, (wink_progress - 0.6) * 2.5)  # Scale to 0-1
            line_y = right_eye_center[1] + int(eye_height * wink_intensity * 0.3)
            draw.line([
                (right_eye_center[0] - eye_width, line_y),
                (right_eye_center[0] + eye_width, line_y)
            ], fill=(255, 255, 255, 220), width=3)
        else:
            # Keep both eyes open (do nothing, original eyes remain)
            pass
        
        frames.append(frame)
        
    return frames

def _apply_displacement(image, displacement):
    """Apply a displacement map to an image."""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create coordinate grids
    height, width = img_array.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply displacement
    new_x = np.clip(x_coords + displacement, 0, width - 1).astype(np.float32)
    new_y = y_coords.astype(np.float32)
    
    # Remap image using OpenCV
    remapped = cv2.remap(
        img_array, 
        new_x, 
        new_y, 
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Convert back to PIL image
    return Image.fromarray(remapped)