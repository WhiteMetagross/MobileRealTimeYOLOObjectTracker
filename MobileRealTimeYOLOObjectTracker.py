#This program implements a real-time object tracking application using YOLOv11 and OSNet for ReID (Re-Identification).
#It captures video from the phone camera, detects objects, tracks them, and uses ReID to match objects across frames.
#The application is built using Kivy for the user interface and ultralytics, and OpenCV for image processing.
#The program is designed to run on Android devices using Pydroid 3, a Python IDE for Android.
#The code includes enhancements for better tracking stability, ReID integration, and performance optimizations.
#The application displays the video feed with detected objects, their IDs, and confidence scores.

import os
import cv2
import numpy as np
from ultralytics import YOLO
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
import threading
import time
from collections import deque
import colorsys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

#Set the environment variables for YOLO configuration directory.
CONFIG_DIR = "/data/data/ru.iiec.pydroid3/files/ultralytics_config"
os.environ["YOLO_CONFIG_DIR"] = CONFIG_DIR
os.makedirs(CONFIG_DIR, exist_ok=True)

#The path to the ByteTrack configuration file.
bytetrack_path = "bytetrack.yaml"

#Check if the ByteTrack configuration file exists.
if not os.path.exists(bytetrack_path):
    exit(1)

#Define the OSNet ReID model class.
#This class handles loading the model, extracting features from images, and computing similarity between features.
class OSNetReID:
    def __init__(self, model_path="osnet_ain_x1_0_imagenet.pth"):
        self.device = torch.device('cpu')
        self.model = None
        self.transform = T.Compose([
            T.Resize((64, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    #Load the OSNet model from a checkpoint file.
    #If the model fails to load, it sets the model to None.
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model = OSNetBackbone()
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.model.to(self.device)
        except:
            self.model = None
    
    #Extract features from a cropped image using the OSNet model.
    #It converts the image to RGB format, applies transformations, and normalizes the features.
    def extract_features(self, crop_img):
        if self.model is None or crop_img is None:
            return None
        
        try:
            if isinstance(crop_img, np.ndarray):
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(crop_img)
            else:
                pil_img = crop_img
            
            input_tensor = self.transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(input_tensor)
                features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
        except:
            return None
    
    #Compute the similarity between two feature vectors.
    #It normalizes the feature vectors and calculates the dot product to determine similarity.
    def compute_similarity(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        
        try:
            feat1 = feat1 / (np.linalg.norm(feat1) + 1e-8)
            feat2 = feat2 / (np.linalg.norm(feat2) + 1e-8)
            return np.dot(feat1, feat2)
        except:
            return 0.0

#Define the OSNet backbone model class.
#This class implements a simplified version of the OSNet architecture for feature extraction.
class OSNetBackbone(torch.nn.Module):
    def __init__(self, num_classes=1000, feature_dim=128):
        super(OSNetBackbone, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        
        self.layer1 = self._make_layer(16, 32, 1)
        self.layer2 = self._make_layer(32, 64, 1, stride=2)
        
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, feature_dim)
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

#Define the basic block for the OSNet architecture.
#This block consists of two convolutional layers with batch normalization and ReLU activation.
class BasicBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                torch.nn.BatchNorm2d(planes)
            )
    
    #Forward pass through the basic block.
    #It applies the convolutional layers, batch normalization, and ReLU activation.
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

try:
    model = YOLO("yolo11m.pt") #Load the YOLOv11 model.
    model.fuse()
    dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
    model(dummy_frame, verbose=False)
    torch.set_num_threads(multiprocessing.cpu_count())
except:
    model = None

reid_model = OSNetReID() #Initialize the OSNet ReID model.

cap = None
latest_frame = None
latest_result = None
result_lock = threading.RLock()
frame_lock = threading.RLock()
processing = False
track_colors = {}
track_memory = {}
frame_queue = deque(maxlen=2)
result_queue = deque(maxlen=2)
track_history = {}
reid_features = {}
reid_gallery = {}
executor = ThreadPoolExecutor(max_workers=3)

#Generate a unique color for each track ID.
#It uses the golden ratio to create a hue value and converts it to RGB format.
def generate_unique_color(track_id):
    if track_id not in track_colors:
        golden_ratio = 0.618033988749895
        hue = (track_id * golden_ratio) % 1.0
        saturation = 0.9
        brightness = 0.9
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
        bgr = tuple(int(c * 255) for c in reversed(rgb))
        track_colors[track_id] = bgr
        
        track_memory[track_id] = {
            'first_seen': time.time(),
            'last_seen': time.time(),
            'confidence_history': deque(maxlen=5),
            'position_history': deque(maxlen=15),
            'stable_count': 0,
            'reid_features': deque(maxlen=2),
            'reid_confidence': 0.0
        }
    
    return track_colors[track_id]

#Update the track memory with the latest detection information.
#It updates the last seen time, confidence history, position history, and stable count.
def update_track_memory(track_id, confidence, position, reid_feature=None):
    if track_id in track_memory:
        track_memory[track_id]['last_seen'] = time.time()
        track_memory[track_id]['confidence_history'].append(confidence)
        track_memory[track_id]['position_history'].append(position)
        track_memory[track_id]['stable_count'] += 1
        
        if reid_feature is not None:
            track_memory[track_id]['reid_features'].append(reid_feature)
            reid_features[track_id] = reid_feature
    else:
        reid_feats = deque(maxlen=2)
        if reid_feature is not None:
            reid_feats.append(reid_feature)
            reid_features[track_id] = reid_feature
        
        track_memory[track_id] = {
            'first_seen': time.time(),
            'last_seen': time.time(),
            'confidence_history': deque([confidence], maxlen=5),
            'position_history': deque([position], maxlen=15),
            'stable_count': 1,
            'reid_features': reid_feats,
            'reid_confidence': 0.0
        }

#Extract a crop from the frame based on the bounding box coordinates.
#It applies a margin to the crop area and ensures the crop is within the frame boundaries.
def extract_crop(frame, box, margin=0.02):
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        margin_w = int(crop_w * margin)
        margin_h = int(crop_h * margin)
        
        x1_crop = max(0, x1 - margin_w)
        y1_crop = max(0, y1 - margin_h)
        x2_crop = min(w, x2 + margin_w)
        y2_crop = min(h, y2 + margin_h)
        
        crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        
        if crop.size > 0 and crop.shape[0] > 8 and crop.shape[1] > 8:
            return crop
        return None
    except:
        return None

#Match the current detections with existing tracks using ReID features.
#It compares the extracted features of the current detections with stored features in the track memory.
def reid_match_tracks(current_detections, frame):
    if reid_model.model is None:
        return current_detections
    
    try:
        for i, detection in enumerate(current_detections):
            if 'crop' in detection and detection['crop'] is not None:
                current_feature = reid_model.extract_features(detection['crop'])
                if current_feature is not None:
                    best_match_id = None
                    best_similarity = 0.0
                    
                    for existing_id, stored_feature in reid_features.items():
                        similarity = reid_model.compute_similarity(current_feature, stored_feature)
                        if similarity > best_similarity and similarity > 0.25:
                            best_similarity = similarity
                            best_match_id = existing_id
                    
                    if best_match_id is not None:
                        track_memory[best_match_id]['reid_confidence'] = best_similarity
        
        return current_detections
    except:
        return current_detections

#Initialize the camera for capturing video frames.
#It sets the camera properties such as FPS, resolution, and exposure.
def initialize_camera():
    global cap
    try:
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
        cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
        
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            return True
        else:
            cap.release()
            cap = None
            return False
    except:
        cap = None
        return False

#Capture frames from the camera in a separate thread.
#It reads frames from the camera, skips every alternative frame, and stores the latest frame in a queue.
def capture_frames():
    global latest_frame, frame_queue
    skip_frames = 0
    
    while cap is not None:
        try:
            ret, frame = cap.read()
            if ret:
                skip_frames += 1
                if skip_frames % 2 == 0:
                    continue
                    
                with frame_lock:
                    latest_frame = frame
                    if len(frame_queue) >= frame_queue.maxlen:
                        frame_queue.popleft()
                    frame_queue.append(frame)
        except:
            break

#Process the captured frames in a separate thread.
#It resizes the frames, runs the YOLO model for object detection, and updates the track memory.
#It also extracts crops for ReID and matches them with existing tracks.
def process_frames():
    global latest_result, processing, result_queue
    frame_counter = 0
    last_process_time = 0
    processing_interval = 0.05
    
    while cap is not None and model is not None:
        try:
            current_time = time.time()
            
            if (len(frame_queue) > 0 and 
                not processing and 
                (current_time - last_process_time) >= processing_interval):
                
                processing = True
                last_process_time = current_time
                
                try:
                    with frame_lock:
                        if frame_queue:
                            process_frame = frame_queue[-1]
                        else:
                            continue
                    
                    original_height, original_width = process_frame.shape[:2]
                    
                    process_size = (416, 416)
                    small_frame = cv2.resize(process_frame, process_size, interpolation=cv2.INTER_LINEAR)
                    
                    results = model.track(
                        source=small_frame,
                        persist=True,
                        tracker=bytetrack_path,
                        conf=0.25, #Confidence threshold for detection.
                        iou=0.5,
                        imgsz=416,
                        verbose=False,
                        half=False,
                        device='cpu',
                        max_det=20, #Maximum number of detections per image.
                        augment=False,
                        agnostic_nms=True,
                        save=False,
                        show=False,
                        stream=False,
                        vid_stride=1
                    )
                    
                    result = results[0] if isinstance(results, list) else results
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        scale_x = original_width / process_size[0]
                        scale_y = original_height / process_size[1]
                        
                        current_detections = []
                        
                        for box in result.boxes:
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id)
                                conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                                x1, y1, x2, y2 = box.xyxy[0]
                                
                                x1_orig = int(x1 * scale_x)
                                y1_orig = int(y1 * scale_y)
                                x2_orig = int(x2 * scale_x)
                                y2_orig = int(y2 * scale_y)
                                
                                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                                
                                crop = None
                                reid_feature = None
                                
                                if frame_counter % 20 == 0:
                                    crop = extract_crop(process_frame, [x1_orig, y1_orig, x2_orig, y2_orig])
                                    if crop is not None:
                                        reid_feature = reid_model.extract_features(crop)
                                
                                current_detections.append({
                                    'track_id': track_id,
                                    'confidence': conf,
                                    'center': center,
                                    'crop': crop,
                                    'reid_feature': reid_feature
                                })
                                
                                update_track_memory(track_id, conf, center, reid_feature)
                        
                        if frame_counter % 20 == 0:
                            reid_match_tracks(current_detections, process_frame)
                    
                    with result_lock:
                        latest_result = result
                        if len(result_queue) >= result_queue.maxlen:
                            result_queue.popleft()
                        result_queue.append(result)
                        
                except:
                    pass
                finally:
                    processing = False
            
            time.sleep(0.001)
            frame_counter += 1
            
        except:
            processing = False
            break

#Define the main Kivy application class.
#This class initializes the user interface, handles camera setup, and processes video frames.
class YOLOApp(App):
    #Initialize the application.
    #It sets the title, icon, and window properties.
    def build(self):
        Window.clearcolor = (0, 0, 0, 1)
        
        layout = BoxLayout(orientation='vertical')
        
        self.status_label = Label(
            text="Initializing enhanced tracking system...",
            size_hint=(1, 0.04),
            color=(0, 1, 0, 1),
            font_size='13sp'
        )
        layout.add_widget(self.status_label)
        
        self.img_widget = Image(
            size_hint=(1, 0.96),
            allow_stretch=True,
            keep_ratio=True
        )
        layout.add_widget(self.img_widget)
        
        self.camera_ready = False
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.detection_fps = 0
        self.last_detection_time = time.time()
        self.detection_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        
        threading.Thread(target=self.init_camera_thread, daemon=True).start()
        
        #Update at 60 FPS. Change as needed.
        Clock.schedule_interval(self.update, 1.0 / 60.0)
        
        return layout
    
    #Initialize the camera and start the capture and processing threads.
    #It checks if the camera is ready and starts the threads for capturing and processing frames.
    def init_camera_thread(self):
        self.camera_ready = initialize_camera()
        if self.camera_ready:
            capture_thread = threading.Thread(target=capture_frames, daemon=True)
            process_thread = threading.Thread(target=process_frames, daemon=True)
            
            capture_thread.start()
            process_thread.start()
            
            reid_status = "ReID Active" if reid_model.model is not None else "ReID Inactive"
            Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', 
                              f"Enhanced Tracking System Online. Display: 60fps | Detection: 20fps | {reid_status}"), 0)

    #Update the application state and display the latest frame with detections.
    #It calculates the FPS, updates the status label, and draws the detections on the frame.
    #It also handles the drawing of bounding boxes, labels, and other information on the frame
    def update(self, dt):
        global latest_frame, latest_result
        
        if not self.camera_ready or latest_frame is None:
            return
        
        try:
            self.frame_count += 1
            
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = current_time
            else:
                self.fps_counter += 1
            
            if len(result_queue) > 0:
                self.detection_count += 1
                if current_time - self.last_detection_time >= 1.0:
                    self.detection_fps = self.detection_count
                    self.detection_count = 0
                    self.last_detection_time = current_time
            
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame
                else:
                    return
            
            with result_lock:
                if latest_result is not None:
                    frame = self.draw_detections_enhanced(frame, latest_result)
            
            if self.frame_count % 20 == 0:
                object_count = 0
                track_count = 0
                memory_tracks = len(track_memory)
                reid_active = len(reid_features)
                
                if latest_result is not None and hasattr(latest_result, 'boxes') and latest_result.boxes is not None:
                    boxes_with_ids = [box for box in latest_result.boxes if hasattr(box, 'id') and box.id is not None]
                    object_count = len(latest_result.boxes)
                    track_count = len(boxes_with_ids)
                    self.active_tracks = track_count
                    self.total_tracks = max(self.total_tracks, memory_tracks)
                
                frame_h, frame_w = frame.shape[:2]
                
                reid_info = f"ReID:{reid_active}" if reid_model.model is not None else "ReID:OFF"
                display_fps = f"Display:{self.current_fps}fps"
                detect_fps = f"Detect:{self.detection_fps}fps"
                objects_info = f"Objects:{object_count}"
                tracks_info = f"Tracks:{track_count}"
                memory_info = f"Memory:{memory_tracks}"
                resolution_info = f"Res:{frame_w}x{frame_h}"
                
                status_parts = [display_fps, detect_fps, objects_info, tracks_info, memory_info, reid_info, resolution_info]
                
                if processing:
                    status_parts.append("Processing")
                
                status_text = " | ".join(status_parts)
                self.status_label.text = status_text
            
            self.display_frame_enhanced(frame)
                
        except:
            pass
    
    #Draw the detections on the frame with bounding boxes, labels, and other information.
    #It scales the bounding box coordinates to match the original frame size and draws the boxes and labels.
    #It also handles the drawing of unique colors for each track ID and displays ReID confidence if available.
    def draw_detections_enhanced(self, frame, result):
        try:
            if not hasattr(result, 'boxes') or result.boxes is None:
                return frame
            
            frame_h, frame_w = frame.shape[:2]
            
            yolo_w, yolo_h = 416, 416
            
            scale_x = frame_w / yolo_w
            scale_y = frame_h / yolo_h
            
            class_names = result.names if hasattr(result, 'names') else {}
            
            base_thickness = max(1, int(frame_w / 600))
            font_scale = max(0.3, frame_w / 3000)
            font_thickness = max(1, int(frame_w / 1500))
            
            for box in result.boxes:
                try:
                    if hasattr(box, 'xyxy'):
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    else:
                        continue
                    
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    class_id = int(box.cls) if hasattr(box, 'cls') else 0
                    class_name = class_names.get(class_id, f"C{class_id}")
                    
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id)
                        color = generate_unique_color(track_id)
                        
                        reid_conf = ""
                        if track_id in track_memory and 'reid_confidence' in track_memory[track_id]:
                            reid_score = track_memory[track_id]['reid_confidence']
                            if reid_score > 0.4:
                                reid_conf = f"R{reid_score:.1f}"
                        
                        label = f"ID{track_id}:{class_name}{reid_conf}"
                        thickness = base_thickness + 1
                    else:
                        color = (160, 160, 160)
                        label = f"{class_name}"
                        thickness = base_thickness
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    center_radius = max(3, int(frame_w / 600))
                    cv2.circle(frame, (cx, cy), center_radius, (0, 0, 255), -1)
                    
                    full_label = f"{label}:{conf:.1f}"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), _ = cv2.getTextSize(full_label, font, font_scale, font_thickness)
                    
                    label_y = y1 - 8 if y1 - 8 > text_height else y2 + text_height + 8
                    
                    cv2.rectangle(frame, 
                                (x1, label_y - text_height - 3), 
                                (x1 + text_width + 3, label_y + 1), 
                                color, -1)
                    
                    cv2.putText(frame, full_label, 
                              (x1 + 1, label_y - 1), 
                              font, font_scale, (255, 255, 255), font_thickness)
                    
                except:
                    continue
            
            return frame
            
        except:
            return frame
    
    #Display the frame with the processed texture.
    #It converts the frame to RGB format, flips it vertically, and creates a texture for the Kivy image widget.
    def display_frame_enhanced(self, frame):
        try:
            h, w = frame.shape[:2]
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            flipped = cv2.flip(rgb_frame, 0)
            
            buf = flipped.tobytes()
            
            texture = Texture.create(size=(w, h), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.img_widget.texture = texture
            
        except:
            pass

    #Handle the application stop event.
    #It releases the camera, clears the track memory, colors, ReID features, and gallery.
    #It also shuts down the executor to stop any running threads.
    def on_stop(self):
        global cap
        
        if cap is not None:
            cap.release()
            cap = None
        
        track_memory.clear()
        track_colors.clear()
        reid_features.clear()
        reid_gallery.clear()
        executor.shutdown(wait=False)

if __name__ == "__main__":
    YOLOApp().run()
