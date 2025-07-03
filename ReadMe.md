# Mobile Real-Time YOLO Object Tracker with Re-Identification

This program is designed to run on Android devices using the Pydroid 3 IDE, leveraging the device's camera for real-time object detection, tracking, and re-identification.

-----

## Overview

The application is a computer vision tool that turns an Android phone into a real-time object tracking device. It captures video directly from the phone's camera, identifies objects in the video stream, assigns a unique ID to each object, and consistently tracks them as they move across frames.

The core of the application is built on a combination of powerful model. It uses **YOLOv11**, a state-of-the-art object detection model, to find objects. For tracking, it employs **ByteTrack**, a robust algorithm that maintains object identities even during temporary occlusions. To enhance tracking stability and re-acquire targets that are lost for longer periods, the system integrates **OSNet**, a deep learning model for Re-Identification (ReID). The user interface is built with **Kivy**, providing a responsive display of the video feed and tracking data.

The entire system is optimized for CPU execution on mobile hardware, using multi-threading to ensure that the UI remains smooth while heavy processing tasks run in the background.

![](./visuals/.gif)
_Output_

-----

## Features

  * **Real-Time Object Detection**: Utilizes the YOLOv11 model to detect a wide range of objects in real-time.
  * **Multi-Object Tracking**: Implements the ByteTrack algorithm to track multiple objects simultaneously, assigning a persistent and unique ID to each one.
  * **Re-Identification (ReID)**: Integrates an OSNet ReID model to re-identify objects after they have been occluded or have left and re-entered the scene, significantly reducing ID switches.
  * **Mobile-First Design**: Specifically engineered to run on Android devices via the Pydroid 3 IDE, with optimizations for mobile CPU performance.
  * **Interactive Kivy GUI**: A clean user interface built with Kivy displays the live camera feed with graphical overlays for tracking information.
  * **Dynamic Visual Feedback**: Each tracked object is highlighted with a uniquely colored bounding box, making it easy to follow individual targets.
  * **Detailed On-Screen Display (OSD)**: The UI provides comprehensive real-time statistics, including display FPS, detection FPS, object and track counts, and ReID status.
  * **Efficient Multi-threaded Architecture**: The application uses separate threads for camera capture and frame processing to prevent the UI from freezing and to maximize throughput.

-----

## How It Works: Algorithms and Methodology

The application's workflow is divided into several key stages, managed by a multi-threaded architecture to ensure real-time performance.

### 1\. Object Detection: YOLOv11

The initial step is to locate objects in each frame. The application uses a pre-trained **YOLOv11** model (`yolo11m.pt`). For performance reasons, each frame from the camera is resized to a smaller resolution (416x416 pixels) before being fed to the model. The detection is configured with a confidence threshold of 0.25 and a maximum of 20 detections per frame to balance accuracy and speed on a mobile device.

### 2\. Object Tracking: ByteTrack

Once objects are detected, they need to be tracked from one frame to the next. This is handled by the **ByteTrack** algorithm, configured via the `bytetrack.yaml` file. ByteTrack is a highly effective tracking-by-detection algorithm notable for its two-stage association process:

  * **High-Confidence Matching**: It first associates high-confidence detection boxes with existing tracks.
  * **Low-Confidence Matching**: Uniquely, it then uses low-confidence detection boxes (which are usually discarded) to recover objects in occluded states, making the tracking much more robust against temporary disappearances.

### 3\. Re-Identification: OSNet

The most advanced feature is the Re-Identification (ReID) system, designed to handle cases where ByteTrack might lose a target (like if a person walks behind a building and reappears later).

  * **Feature Extraction**: The `OSNetReID` class uses a lightweight OSNet (Omni-Scale Network) model (`osnet_ain_x1_0_imagenet.pth`) to analyze the visual appearance of a detected object. It extracts a compact feature vector (an "embedding") that represents the unique visual signature of that object.
  * **Gallery of Features**: The system maintains a gallery (`reid_features`) that stores the feature vectors for recently tracked objects.
  * **Similarity Matching**: When a new object is detected, its features are extracted and compared against the features in the gallery using cosine similarity. If the similarity score is high enough (above a threshold of 0.25), the system concludes that the new detection is the same object as one seen before and assigns it the correct, pre-existing track ID. This prevents the system from creating a new, incorrect ID.

### 4\. Application Architecture

The program's architecture is designed for concurrency and responsiveness.

  * **Main Thread (Kivy)**: The `YOLOApp` class runs on the main thread, handling all UI elements, rendering the final video frame, and displaying status text.
  * **Capture Thread (`capture_frames`)**: This dedicated thread's only job is to communicate with the camera hardware. It continuously captures frames and places them in a shared queue (`frame_queue`), ensuring the processing thread always has the most recent image to work with.
  * **Processing Thread (`process_frames`)**: This is the computational core. It grabs frames from the queue, runs the YOLOv11 detection and ByteTrack tracking, and periodically triggers the OSNet ReID feature extraction and matching. By offloading this intensive work, the UI remains fluid at 60 FPS, even though the detection itself may run slower (e.g., 20 FPS).

-----

## File & Folder Structure

For the application to work, your project folder should be structured as follows. You will need to download the model weight files separately.

```
MobileRealTimeYOLOObjectTracker\
├── MobileRealTimeYOLOObjectTracker.py  #Main Python application program.
├── bytetrack.yaml                      #Configuration file for the tracker.
├── requirements.txt                    #List of Python dependencies.
├── yolo11m.pt                          #YOLOv11 model weights (must be downloaded).
└── osnet_ain_x1_0_imagenet.pth         #OSNet ReID model weights (must be downloaded).
```

-----

## Installation and Usage on Pydroid 3

Follow these steps to set up and run the project on your Android device.

### Download the pretrained models:
  1. **YOLOv11**: https://docs.ultralytics.com/models/yolo11/#performance-metrics
  2. **OSNet ReID**: https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md

### Prerequisites

  * Install the **Pydroid 3 - IDE for Python 3** from the Google Play Store.
  * Grant Pydroid 3 storage and camera permissions when prompted.
  * Download and place `yolo11m-obb.pt` in the root directory. This is the default model path used by the application.
  * Download `osnet_ain_x1_0_imagenet.pth` from the [fast-reid MODEL\_ZOO](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md). Place the model in the root directory. This is the default model path used by the application.

### Step 1: Get Project Files

Download all the files from this repository (`.py`, `.yaml`, `.txt`) and place them together in a single folder on your Android device's internal storage.

### Step 2: Download Model Weights

You must obtain the model weights for YOLOv11 and OSNet.

1.  **YOLOv11 Model**: Download the `yolo11m.pt` file.
2.  **OSNet Model**: Download the `osnet_ain_x1_0_imagenet.pth` file.

Place both of these `.pt` and `.pth` files in the **same folder** as the `MobileRealTimeYOLOObjectTracker.py` script.

### Step 3: Install Python Dependencies

Pydroid 3 uses the Pip package manager. You must install the required libraries.

1.  Open Pydroid 3, go to the menu (☰) and select **Pip**.

2.  Select the **LIBRARIES** tab and install each of the following packages one by one. For PyTorch, you will need to use a specific command.

      * `kivy`
      * `ultralytics`
      * `opencv-python`
      * `numpy`
      * `pillow`

3.  To install **PyTorch**, go to the **QUICK INSTALL** tab. In the list of popular packages, find and install **torch**. If that fails, go back to the menu, select **Terminal**, and type the following command:
    `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu`

### Step 4: Run the Application

1.  In Pydroid 3, use the file navigator to open the `MobileRealTimeYOLOObjectTracker.py` script.
2.  Press the large yellow "Play" button at the bottom right to run the program.
3.  The application will start, and after a brief initialization, you will see the live feed from your camera with object tracking overlays.

-----

## Understanding the On-Screen Display

The application provides a rich set of information overlaid on the video feed.

  * **Bounding Boxes**: Each detected object is enclosed in a rectangle. The color of the rectangle is unique to the object's track ID.
  * **Object Label**: Above the bounding box, you will find a label with detailed information:
      * **ID{id}**: The unique tracking ID assigned by ByteTrack.
      * **{class\_name}**: The class of the object detected by YOLOv11.
      * **R{score}**: An optional Re-Identification confidence score. This only appears if OSNet successfully re-identifies a track, indicating the confidence of the match.
      * **{conf}**: The detection confidence score from YOLO, ranging from 0.0 to 1.0.
  * **Status Bar**: A status line at the top provides real-time performance metrics, separated by `|`:
      * **Display**: The frames-per-second (FPS) of the user interface rendering. Aims for 60 FPS.
      * **Detect**: The FPS of the underlying detection and tracking model. This will be lower than the display FPS.
      * **Objects**: The total number of objects detected in the current frame.
      * **Tracks**: The number of detected objects that have been assigned a stable track ID.
      * **Memory**: The total number of unique object IDs currently held in the tracker's memory.
      * **ReID**: The number of tracks for which a ReID feature vector has been generated and stored. Shows "ReID:OFF" if the model failed to load.
      * **Res**: The resolution (Width x Height) of the video feed being processed.
