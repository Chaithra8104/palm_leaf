# palm_leaf
Digitization and semantic translation system for Kannada palm-leaf manuscripts, converting Halegannada characters into modern Kannada and English using image processing, CNN-based recognition, and language translation techniques.


# Digitization and Semantic Translation of Kannada Palm Leaf Manuscripts

## 📌 Project Overview

This project focuses on the digitization and semantic translation of historical Kannada palm-leaf manuscripts. The system processes images of palm-leaf manuscripts, extracts individual characters using image preprocessing and segmentation techniques, recognizes Halegannada characters using CNN-based OCR, and translates the recognized text into modern Kannada and English.

The project aims to preserve historical Kannada manuscripts digitally while making their content more accessible to modern readers.

## 🚀 Key Features

- Digital image processing of Kannada palm-leaf manuscripts
- Image preprocessing and noise reduction
- Character and glyph segmentation using OpenCV
- CNN-based character recognition
- OCR-based extraction of Halegannada text
- Halegannada → modern Kannada semantic translation
- Modern Kannada → English translation
- Dictionary-based and AI-assisted translation
- Interactive interface for testing the OCR and translation pipeline

## 🔄 System Workflow

Palm Leaf Image  
↓  
Image Preprocessing  
↓  
Character Segmentation  
↓  
CNN-based Character Recognition  
↓  
Halegannada Text Generation  
↓  
Semantic Translation  
↓  
Modern Kannada  
↓  
English Translation

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Image Processing:** OpenCV
- **Machine Learning:** CNN
- **OCR:** Character Recognition
- **Translation:** Dictionary-based and AI-assisted translation
- **Interface:** Streamlit

## 🎯 Objective

The primary objective of this project is to build a computational pipeline for preserving and understanding historical Kannada palm-leaf manuscripts by converting handwritten/inscribed Halegannada characters into digitally readable text and translating their meaning into modern Kannada and English.

## 📊 Dataset

The project includes a custom dataset of Kannada manuscript characters consisting of:

- 6,143 base character samples
- 2,057 vowel modifier samples
- 2,170 ottakshara samples
- **10,370 total character samples**

A Halegannada-to-Hosagannada dictionary containing approximately **31,485 entries** is also used for the translation stage.

## 🔮 Future Scope

- Improve recognition accuracy for complex and damaged characters
- Expand the manuscript dataset
- Support additional historical Kannada scripts
- Improve contextual and semantic translation
- Develop a complete web-based manuscript digitization platform
