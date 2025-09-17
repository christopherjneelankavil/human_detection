import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
from tensorflow.lite.python.interpreter import Interpreter as tflite
from gpiozero import OutputDevice, PWMOutputDevice

# Define motor control pins (fixed as per your original code)
# Left motor
IN3 = OutputDevice(12)   # Left motor forward
IN4 = OutputDevice(16)   # Left motor backward
IN1 = OutputDevice(20)   # Left motor forward
IN2 = OutputDevice(21)   # Left motor backward
ENB = PWMOutputDevice(18)  # Left motor speed (PWM)
ENB2 = PWMOutputDevice(13) # Left motor speed (PWM)

# Right motor
IN1_2 = OutputDevice(6)   # Right motor forward
IN2_2 = OutputDevice(5)   # Right motor backward
IN3_2 = OutputDevice(22)  # Right motor forward
IN4_2 = OutputDevice(27)  # Right motor backward
ENA_2 = PWMOutputDevice(23) # Right motor speed (PWM)
ENB_2 = PWMOutputDevice(24) # Right motor speed (PWM)

# Initial speed
speed = 0.7
ENB.value = speed
ENB2.value = speed
ENA_2.value = speed
ENB_2.value = speed

# Configuration for detection
cap = cv2.VideoCapture(0)
threshold = 0.3
top_k = 5
model_dir = './models'
model_file = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
label_file = 'coco_labels.txt'

# Global variables for robot control
tolerance = 0.1
x_deviation = 0
y_max = 0
bbox_area = 0
frame_area = 0
area_threshold = 0.85  # Stop when bbox covers 85% of frame
object_to_track = 'person'

# Motor control functions
def move_forward():
    IN3.on()    # Left motor forward
    IN4.off()
    IN1.off()
    IN2.on()
    IN1_2.on()  # Right motor forward
    IN2_2.off()
    IN3_2.off()
    IN4_2.on()

def move_backward():
    IN3.off()   # Left motor backward
    IN4.on()
    IN1.on()
    IN2.off()
    IN1_2.off() # Right motor backward
    IN2_2.on()
    IN3_2.on()
    IN4_2.off()

def rotate_left(duration):
    IN3.off()   # Left motor backward
    IN4.on()
    IN1.on()
    IN2.off()
    IN1_2.on()  # Right motor forward
    IN2_2.off()
    IN3_2.off()
    IN4_2.on()
    time.sleep(duration)
    stop_motor()

def rotate_right(duration):
    IN3.on()    # Left motor forward
    IN4.off()
    IN1.off()
    IN2.on()
    IN1_2.off() # Right motor backward
    IN2_2.on()
    IN3_2.on()
    IN4_2.off()
    time.sleep(duration)
    stop_motor()

def stop_motor():
    IN3.off()
    IN4.off()
    IN1.off()
    IN2.off()
    IN1_2.off()
    IN2_2.off()
    IN3_2.off()
    IN4_2.off()

# Load the TFLite model and labels
def load_model(model_dir, model_file, label_file):
    model_path = f"{model_dir}/{model_file}"
    label_path = f"{model_dir}/{label_file}"
    
    interpreter = tflite(model_path=model_path)
    interpreter.allocate_tensors()
    
    with open(label_path, 'r') as f:
        labels = {i: line.strip() for i, line in enumerate(f.readlines())}
    
    return interpreter, labels

# Set —

def set_input(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    height = input_details['shape'][1]
    width = input_details['shape'][2]
    
    resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
    input_data = np.expand_dims(np.array(resized_image), axis=0)
    
    if input_details['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (input_data / 255.0).astype(np.float32)
    
    interpreter.set_tensor(input_details['index'], input_data)

# Get output from the model
def get_output(interpreter, score_threshold, top_k):
    output_details = interpreter.get_output_details()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
    
    detections = []
    for i in range(min(num_detections, top_k)):
        if scores[i] > score_threshold:
            detections.append({
                'bbox': boxes[i],
                'class': int(classes[i]),
                'score': scores[i]
            })
    return detections

# Draw bounding boxes and count persons
def draw_boxes(frame, detections, labels):
    height, width = frame.shape[:2]
    person_count = 0
    
    for detection in detections:
        class_id = detection['class']
        if class_id != 0:  # Class ID 0 is "person"
            continue
        
        person_count += 1
        ymin, xmin, ymax, xmax = detection['bbox']
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        text = f"person: {detection['score']:.2f}"
        cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    detection_text = f"Persons: {person_count}"
    text_size = cv2.getTextSize(detection_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = width - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, detection_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return person_count

# Tracking logic with area calculation
def track_object(detections, labels, frame_height, frame_width):
    global x_deviation, y_max, tolerance, bbox_area, frame_area
    
    frame_area = frame_height * frame_width
    
    if len(detections) == 0:
        print("No objects to track")
        stop_motor()
        return
    
    flag = 0
    for detection in detections:
        class_id = detection['class']
        if class_id == 0:  # "person"
            ymin, xmin, ymax, xmax = detection['bbox']
            flag = 1
            break
    
    if flag == 0:
        print("Selected object not present")
        stop_motor()
        return
    
    # Convert to pixel values
    xmin_px = int(xmin * frame_width)
    xmax_px = int(xmax * frame_width)
    ymin_px = int(ymin * frame_height)
    ymax_px = int(ymax * frame_height)
    
    # Calculate bounding box area
    bbox_width = xmax_px - xmin_px
    bbox_height = ymax_px - ymin_px
    bbox_area = bbox_width * bbox_height
    
    # Calculate center point
    x_diff = xmax - xmin
    obj_x_center = xmin + (x_diff / 2)
    x_deviation = round(0.5 - obj_x_center, 3)
    y_max = round(ymax, 3)
    
    print("{", x_deviation, y_max, "area_ratio:", bbox_area/frame_area, "}")
    thread = Thread(target=move_robot)
    thread.start()

# Area-based movement logic
def move_robot():
    global x_deviation, y_max, tolerance, bbox_area, frame_area, area_threshold
    
    # Calculate area ratio
    area_ratio = bbox_area / frame_area
    
    # Debug info
    print(f"Debug - x_deviation: {x_deviation}, area_ratio: {area_ratio:.4f}")
    
    # First check if person is too close based on area
    if area_ratio >= area_threshold:
        stop_motor()
        print(f"Person too close! Area ratio: {area_ratio:.4f} >= {area_threshold}")
        return
    
    # If not too close, handle turning based on x_deviation
    if abs(x_deviation) < tolerance:
        # Person is centered and not too close, move forward
        move_forward()
        print("Moving forward - person centered and not too close")
    elif x_deviation > 0:  # Person is left of center
        delay1 = get_delay(x_deviation)
        rotate_left(delay1)
        print(f"Turned left for {delay1:.3f} seconds - x_deviation: {x_deviation}")
    elif x_deviation < 0:  # Person is right of center
        delay1 = get_delay(x_deviation)
        rotate_right(delay1)
        print(f"Turned right for {delay1:.3f} seconds - x_deviation: {x_deviation}")

# Delay calculation
def get_delay(deviation):
    deviation = abs(deviation)
    if deviation >= 0.4:
        d = 0.080
    elif deviation >= 0.35 and deviation < 0.40:
        d = 0.060
    elif deviation >= 0.20 and deviation < 0.35:
        d = 0.050
    else:
        d = 0.040
    return d

def main():
    interpreter, labels = load_model(model_dir, model_file, label_file)
    
    fps = 1
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        frame_height, frame_width = frame.shape[:2]
        
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            pil_im = Image.fromarray(cv2_im)
        except Exception as e:
            print(f"Error converting frame to PIL image: {e}")
            continue
        
        set_input(interpreter, pil_im)
        interpreter.invoke()
        detections = get_output(interpreter, score_threshold=threshold, top_k=top_k)
        
        person_count = draw_boxes(frame, detections, labels)
        track_object(detections, labels, frame_height, frame_width)
        
        fps = round(1.0 / (time.time() - start_time), 1)
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Person Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    stop_motor()

if __name__ == '__main__':
    main()