# Multi_Images

🐟 Multiclass Fish Image Classification – Deep Learning Project

The growing importance of automated species identification in fisheries, marine research, and food industries has led to an increasing need for accurate and scalable image-based classification systems. With modern advancements in Deep Learning and transfer learning, it is now possible to classify fish species with high precision, even in complex underwater or market environments.

This project aims to create a state-of-the-art fish classification system using CNN models, transfer learning architectures, and interactive Streamlit deployment — enabling real-world usage such as fisheries automation, quality control, and mobile-based fish species recognition.

Licensed for educational and open-source use, this project demonstrates a full end-to-end AI workflow: data preprocessing → model training → model comparison → deployment.

📢 Announcements

🌟 EfficientNetB0 achieved 87% validation accuracy — the best-performing model so far.
🌟 Streamlit interface ready for deployment (image upload → real-time prediction).
🌟 Full documentation, README, and project report generated.

📑 Table of Contents

Fish Image Classification – Project Overview

Announcements

Table of Contents

Goal

Guide

Documentation

Folder Structure

Dataset Structure & Syntax

Data Loading

Data Augmentation

Preprocessing

Model Architectures

CNN from Scratch

Transfer Learning Models

Evaluation Metrics

Streamlit Application

FAQs

License

🎯 Goal

Our goal is to build a robust, scalable deep learning system capable of classifying fish images into multiple species with high accuracy. Using both CNN and transfer learning models, this project enables learners and developers to:

Understand how image classification pipelines work

Compare models and evaluate accuracy

Deploy the best model using a simple, interactive web interface

Use the model for real-world applications such as market automation, fish recognition, and research

📘 Guide

This project is structured to provide a clear understanding of each phase of the computer vision pipeline:

1️⃣ Data Preprocessing

Loading the dataset using ImageDataGenerator

Rescaling images

Splitting train/validation sets

Applying augmentation

2️⃣ Model Building

Train a CNN model from scratch

Train 5 transfer-learning models:

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0

3️⃣ Model Evaluation & Selection

Compare:

Accuracy

Loss

Classification report

Confusion matrix

Select best model

Save .h5 file

4️⃣ Deployment

Build Streamlit interface

Upload image → Get prediction + confidence scores

Display top model info

📁 Folder Structure
fish-classification/
|___ data/
|    |___ Salmon/
|    |___ Tuna/
|    |___ Trout/
|    |___ Mackerel/
|    |___ Sardine/
|
|___ models/
|    |___ cnn_model.h5
|    |___ efficientnetb0_best.h5
|    |___ resnet50.h5
|
|___ streamlit_app/
|    |___ app.py
|    |___ model_loader.py
|
|___ notebooks/
|    |___ training.ipynb
|    |___ evaluation.ipynb
|
|___ reports/
|    |___ project_report.pdf
|
|___ README.md
|___ requirements.txt

🐟 Dataset Structure / Syntax

The dataset is organized into separate directories — each representing a fish species.

data/
|___ Salmon/
|___ Tuna/
|___ Trout/
|___ Mackerel/
|___ Sardine/

✔ Supported Image Formats

JPG

PNG

✔ Labels

Assigned automatically from folder names.

✔ Example Flow

Salmon/xxx.jpg → Label = "Salmon"

🧠 Model Training
1️⃣ CNN From Scratch

3 Conv2D + MaxPooling layers

Dense layers with dropout

Output layer: Softmax (5 classes)

2️⃣ Transfer Learning Models

All models pretrained on ImageNet:

Model	Status	Accuracy
VGG16	Completed	81%
ResNet50	Completed	84%
MobileNet	Completed	85%
InceptionV3	Completed	83%
EfficientNetB0	⭐ Best Model	87%

The EfficientNetB0 model was selected as the final deployed model.

📊 Evaluation Metrics

Accuracy

Validation accuracy

Loss curves

Classification report

Confusion matrix

Model comparison chart

🌐 Streamlit Application
Features:

✔ Upload fish image
✔ Predict species
✔ Show confidence score
✔ Display sampled dataset images
✔ Load best model automatically

Start App:
streamlit run app.py

❓ FAQs
1. How many classes are supported?

Five: Salmon, Tuna, Trout, Mackerel, Sardine

2. Can this be deployed online?

Yes — using Streamlit Cloud or AWS EC2.

3. Can the model be retrained?

Yes. Training notebooks are included.

📜 License

Open-source under MIT License — free to use, modify, and distribute.
