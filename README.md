# Hand Gesture Controlled Game (CNN + MediaPipe)

## Overview

This project implements a real-time hand gesture recognition system that controls a 2D game using computer vision and deep learning.

The system includes:
- Custom gesture data collection
- CNN model training and evaluation
- Baseline comparison (SVM and MLP)
- Real-time gesture prediction
- Integration with a PyGame-based game environment

Five gesture classes were defined:
- fist -> maps to attack in-game
- point_left -> walk left
- point_right -> walk right
- point_up -> jump
- palm -> pause


## Repository Structure

sample_images/ – Sample preprocessed gesture images  
spritesheets/ – Provides a default sprite for the game. Sprites for demo was not included as those were personal assets
game.py – PyGame game loop  
live_prediction_keys.py – Real-time CNN prediction and keyboard mapping  
record_gestures.py – Gesture data collection script  
train_cnn.py – CNN training  
train_baseline.py – Baseline model training (SVM / MLP)  
test_cnn.py – CNN evaluation  
test_baseline.py – Baseline evaluation  
requirements.txt – Python dependencies  

---

## Installation

1. Create a virtual environment:
python -m venv venv
venv\Scripts\activate
2. Install Dependencies:
pip install -r requirements.txt

## Prerequesites
1. Require CNN model.
CNN Model in drive as the file is too large to be uploaded in Github.
Download from:  https://drive.google.com/drive/folders/1H1SRBnbPzsQLneQQ3KYWVI4Zr5zVXC_0?usp=sharing
2. Sprite Asset.
 Require spritesheets/soldier.png 


## Running the Project
Start real-time gesture prediction:
python live_prediction_keys.py
Then start the game:
python game.py

Ensure webcam is enabled, once both windows are open, gestures can be used to control the game.

## Environment

Python 3.10  
TensorFlow 2.15.0  
MediaPipe 0.10.14  
OpenCV 4.11.0  
NumPy 1.26.4  
Pandas 2.3.3  
scikit-learn 1.7.2  
Pygame 2.6.1  

## Notes

- The full dataset is not included due to size limitations. Sample images are provided for references.
- Models were trained on a self-collected dataset captured across multiple sessions and environments.
- soldier.png sprite is sourced from the Littlewargame and are used here for demonstration purposes only.

