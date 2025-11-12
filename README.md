# Deepfake Detection Using MiniFASNetV2

This project implements a lightweight and efficient deepfake detection system using the MiniFASNetV2 face anti-spoofing model. Instead of detecting identity mismatches, the system analyzes subtle texture, noise, and physiological inconsistencies found in manipulated faces to distinguish real videos from deepfakes.

## Key Features
- Uses MiniFASNetV2, a lightweight CNN designed for face anti-spoofing.
- Detects deepfakes based on texture-level inconsistencies rather than identity comparison.
- Frame sampling at 3 FPS for efficient video analysis.
- Reliable face detection using MTCNN (facenet-pytorch).
- Median aggregation of frame-level predictions for robust final outputs.
- Supports GPU (CUDA) and CPU execution.

## Pipeline Overview
1. Frame Sampling  
   Extracts frames from the input video at 3 FPS.

2. Face Detection (MTCNN)  
   Localizes and aligns faces within each sampled frame.

3. Face Preprocessing  
   Crops and resizes detected faces to 80×80, applies normalization, and converts to tensor format.

4. Anti-Spoofing Inference (MiniFASNetV2)  
   Produces real/spoof probabilities based on facial texture patterns.

5. Final Classification  
   Aggregates frame-level predictions using the median to classify the entire video as REAL or FAKE.

## Requirements

To install dependencies:
pip install -r requirements.txt


Main libraries used:
- PyTorch  
- facenet-pytorch  
- OpenCV  
- NumPy  
- tqdm  
- colorama  

## Usage

Run inference on a video:
python infer_demo.py samples/real.mp4


The output includes:
- Number of processed frames  
- Fake probability score (median)  
- Final classification (REAL or FAKE)


## Model Information

MiniFASNetV2 is a compact convolutional architecture originally developed for face anti-spoofing. Its ability to detect fine-grained artifacts such as texture irregularities, blending defects, and missing micro-patterns makes it effective for identifying deepfakes.


