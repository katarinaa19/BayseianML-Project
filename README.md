# **Image Captioning**  
📅 **Date:** 3/7/2025

👥 **Team Members:** Terry Zhuang, Yijia Song, Yunlei Xu, Katarina Wang

---

## **🌟1. Overview**  

### **🎯Objective**  
The objective of this project is to develop an Image Captioning model that can generate accurate and meaningful textual descriptions for images. By leveraging Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory networks (LSTMs) for text generation, the model aims to:
- **Process Textual and Visual Data**: Tokenize and preprocess captions to create a structured vocabulary for training, while simultaneously extracting image features using a pre-trained CNN (e.g., DenseNet201) to encode visual content into feature vectors.  
- **Train a Sequence-to-Sequence Model**: Combine CNN-extracted features with LSTM-based text generation to predict the next word in a caption.
- **Generate Captions for New Images**: Use the trained model to describe unseen images with relevant and coherent text.

### **📂Dataset**  
- **Data Description**: The Flickr8k Dataset consists of 8,091 images, each paired with five unique textual descriptions, totaling 40,455 captions. It is specifically designed for training and evaluating models that generate natural language descriptions for images.
- **Images:** 8,091 natural scene images.
- **Captions per image:** Each image has 5 corresponding descriptions.
- **Source:** https://www.kaggle.com/datasets/adityajn105/flickr8k
![image](https://github.com/user-attachments/assets/8a8360c4-7803-44e8-b0c5-109a95f16702)

---

## **🔄2. Project Workflow**  

### **Model Workflow**  
![image](https://github.com/user-attachments/assets/c0593782-75c7-47bb-85ca-e81d1aff9b02)


### **📌 Step 1: Data Preprocessing**  
#### 1.1 Preprocessing Captions 
- Normalization
  - Converts text to lowercase for consistency.
  - Removes punctuation, special symbols, and numbers.
  - Eliminates extra spaces and single-letter words.
  - Adds "startseq" and "endseq" to mark sequence boundaries.

```python
captions = data_first_1000['caption'].str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True).apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
captions = "startseq " + captions + " endseq"
```
- Tokenization
  - Initializes a Tokenizer to convert words into numerical tokens.
  - Uses oov_token="<OOV>" for out-of-vocabulary words.
  - Trains on cleaned captions, assigning unique indices to words.
  
```python
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(captions)
```
- Define Vocabulary and Caption Length
  - vocab_size: Total unique words from the tokenizer plus one for padding.
  - max_caption_length: Length of the longest caption, used for padding.

```python
vocab_size = len(tokenizer.word_index) + 1
max_caption_length = max(len(caption.split()) for caption in captions)
```
- Convert Captions to Sequences
  - Converts each caption into a sequence of integers, where each word is replaced by its corresponding token index from the tokenizer.
  
```python
sequences = tokenizer.texts_to_sequences(captions)
```
- Padding the Sequences
  - Ensures that all sequences have the same length (max_caption_length).
  - Uses post-padding, meaning zeros are added at the end of shorter captions.

```python
padded_sequences = pad_sequences(sequences, maxlen=max_caption_length, padding='post')
```

#### 1.2 Preprocessing Images
#### 1.3 Train Test Split



### **📌 Step 2: Encode**  
#### 2.1 Image Features Encoding with CNNs
- Extract image features using `DenseNet201` to convert images into meaningful numerical vectors.  
- Reduce dimensionality using `Dense(256)` to ensure a compact representation of the image.  
- Convert features to 3D using `RepeatVector(max_caption_length-1)`
  - `(256,) → (max_caption_length-1, 256)`
  - This ensures image features align with textual input sequences.
  
```python
input1 = Input(shape=(X_train.shape[1],))
img_features = Dense(embedding_dim, activation='relu')(input1)
img_features = RepeatVector(max_caption_length-1)(img_features)  # Convert to 3D
```

#### 2.2 Text Features Encoding with LSTMs
- Convert words into vector representations using an `Embedding` layer.  
- Transform text into 3D format, which is required for sequential processing in LSTM.

```python
input2 = Input(shape=(max_caption_length-1,))
text_features = Embedding(vocab_size, embedding_dim, mask_zero=True)(input2)
text_features = LSTM(lstm_units, return_sequences=True)(text_features)
```


### **📌 Step 3: Decode**  
#### 3.1 Concatenate Captions and Images to Fit in LSTM Model  
- Two-step merging process:  
  1. `Add()` merges visual and textual features.  
  2. LSTM processes the combined sequence, learning relationships between image content and text.  

```python
decoder = Add()([img_features, text_features])
decoder = LSTM(lstm_units, return_sequences=True)(decoder)
```

#### 3.2 Use a Dense + LSTM-based Decoder to Predict the Next Word in the Sequence
- LSTM generates the caption word by word, predicting the next word based on the image and previous words.  

#### 3.3 Model Refinement
- Use two `Dense` layers to extract key features and improve accuracy.  
- Apply a Softmax layer to select the most probable word from the vocabulary.  
- Use Dropout (0.5) to prevent overfitting and improve generalization.  

```python
decoder = Dense(output_layer_1, activation='relu')(decoder)
decoder = Dense(output_layer_2, activation='relu')(decoder)
decoder = Dropout(0.5)(decoder)  # Prevent overfitting
output = Dense(vocab_size, activation='softmax')(decoder)
```

---

## ***📊3. Model Results & Testing**  
- [Explain how the model is evaluated]  
- [Metrics used: BLEU, ROUGE, or others]  
- [Show sample results: image + generated caption]  
