# Face-Mask-Detection

**Project Overview**

This project detects whether a person is wearing a face mask or not using a combination of:

MTCNN (Multi-task Cascaded Convolutional Networks) for face detection
Convolutional Neural Network (CNN) for mask classification
OpenCV for image processing
TensorFlow/Keras for deep learning

The model first detects faces in an image and then classifies each detected face as:

->Face With Mask
->Face Without Mask
->Features
Face detection using MTCNN
Face mask classification using CNN
Image preprocessing and augmentation
Training and evaluation pipeline
Prediction on unseen test images
Submission file generation for Kaggle competitions

**Dataset**

Dataset contains:

Images with faces
Bounding box annotations
Labels:
face_with_mask
face_no_mask

Example Dataset Structure:

Medical Mask/
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── annotations/
│
├── train.csv
└── submission.csv


 **Technologies Used**
 1.Technology	Purpose
 2.Python	Programming Language
 3.TensorFlow/Keras	Deep Learning
 4.OpenCV	Image Processing
 5.MTCNN	Face Detection
 6.NumPy	Numerical Operations
 7.Pandas	Data Handling
 8.Matplotlib	Visualization
 9.Installation

**Clone the repository:**

git clone https://github.com/yourusername/face-mask-detection.git

cd face-mask-detection

**Install dependencies:**

pip install tensorflow
pip install opencv-python
pip install mtcnn
pip install numpy pandas matplotlib tqdm


**Model Architecture**

The CNN architecture consists of:

Input Image (50x50x1)
        │
Conv2D (100 filters)
        │
MaxPooling2D
        │
Conv2D (64 filters)
        │
MaxPooling2D
        │
Flatten
        │
Dense (50 neurons)
        │
Dropout (0.2)
        │
Dense (2 neurons)
        │
Softmax
 Training

Compile the model:

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

Train the model:

model.fit(
    x,
    y,
    epochs=5,
    batch_size=5
)
 Face Detection

Faces are detected using MTCNN:

from mtcnn import MTCNN

detector = MTCNN()
faces = detector.detect_faces(image)

Each detected face is cropped and passed to the CNN classifier.

**Results**


The model can:

Detect faces from images
Classify mask status
Generate predictions for test datasets
Export results into CSV format

Example Output:

Image Name          Prediction
--------------------------------
img_001.jpg         Face With Mask
img_002.jpg         Face No Mask
img_003.jpg         Face With Mask
** Project Structure**
face-mask-detection/
│
├── face_mask_detection.ipynb
├── train.csv
├── submission.csv
├── images/
├── annotations/
│
├── README.md
└── requirements.txt


 **Future Improvements**
Improve model accuracy using Transfer Learning
Use MobileNetV2 or EfficientNet
Real-time webcam mask detection
Deployment using Flask/Streamlit
Support multiple face detection models
