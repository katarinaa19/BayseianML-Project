# **Image Captioning**  
📅 **Date:** 3/7/2025

👥 **Team Members:** Terry Zhuang, Yijia Song, Yunlei Xu, Katarina Wang

---

## **1. Overview**  

### **Objective**  
The objective of this project is to develop an Image Captioning model that can generate accurate and meaningful textual descriptions for images. By leveraging Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory networks (LSTMs) for text generation, the model aims to:
- **Process Textual and Visual Data**: Tokenize and preprocess captions to create a structured vocabulary for training, while simultaneously extracting image features using a pre-trained CNN (e.g., DenseNet201) to encode visual content into feature vectors.  
- **Train a Sequence-to-Sequence Model**: Combine CNN-extracted features with LSTM-based text generation to predict the next word in a caption.
- **Generate Captions for New Images**: Use the trained model to describe unseen images with relevant and coherent text.

### **Dataset**  
- **Data Description**: The Flickr8k Dataset consists of 8,091 images, each paired with five unique textual descriptions, totaling 40,455 captions. It is specifically designed for training and evaluating models that generate natural language descriptions for images.
- **Images:** 8,091 natural scene images.
- **Captions per image:** Each image has 5 corresponding descriptions.
- **Source:** https://www.kaggle.com/datasets/adityajn105/flickr8k

---

## **2. Project Workflow**  

### **Model Workflow**  
![Workflow Diagram](https://github.com/user-attachments/assets/50c75e10-c497-4581-ba44-684b9337360a)  

### **📌 Step 1: Data Preprocessing**  
- [Brief explanation of preprocessing]  

### **📌 Step 2: Encode**  
- [Brief explanation of encoding process]  

### **📌 Step 3: Model Fitting**  
- [Brief explanation of training the model]  

### **📌 Step 4: Decode**  
- [Brief explanation of how captions are generated]  

---

## **3. Model Results & Testing**  
- [Explain how the model is evaluated]  
- [Metrics used: BLEU, ROUGE, or others]  
- [Show sample results: image + generated caption]  
