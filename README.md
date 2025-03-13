# **Image Captioning**  
📅 **Date:** 3/7/2025

👥 **Team Members:** Terry Zhuang, Yijia Song, Yunlei Xu, Katarina Wang

---

## **🌟 1. Overview**  

### **🎯 Objective**  
The objective of this project is to develop an Image Captioning model that can generate accurate and meaningful textual descriptions for images. By leveraging Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory networks (LSTMs) for text generation, the model aims to:
- **Process Textual and Visual Data**: Tokenize and preprocess captions to create a structured vocabulary for training, while simultaneously extracting image features using a pre-trained CNN (e.g., DenseNet201) to encode visual content into feature vectors.  
- **Train a Sequence-to-Sequence Model**: Combine CNN-extracted features with LSTM-based text generation to predict the next word in a caption.
- **Generate Captions for New Images**: Use the trained model to describe unseen images with relevant and coherent text.

### **📂 Dataset**  
- **Data Description**: The Flickr8k Dataset consists of 8,091 images, each paired with five unique textual descriptions, totaling 40,455 captions. It is specifically designed for training and evaluating models that generate natural language descriptions for images.
- **Images:** 8,091 natural scene images.
- **Captions per image:** Each image has 5 corresponding descriptions.
- **Source:** https://www.kaggle.com/datasets/adityajn105/flickr8k
![image](https://github.com/user-attachments/assets/8a8360c4-7803-44e8-b0c5-109a95f16702)

---

## **🔄 2. Model Workflow**  

### **Flowchart**  
![image](https://github.com/user-attachments/assets/0555ff8f-a24a-41a8-acd2-65b31b388634)


### **📌 Step 1: Data Preprocessing**  
#### 1.1 Preprocessing Captions 
- Normalization
  - Converts text to lowercase for consistency.
  - Removes punctuation, special symbols, and numbers.
  - Eliminates extra spaces and single-letter words.
  - Adds "startseq" and "endseq" to mark sequence boundaries.

```python
captions = data['caption'].str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True).apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
captions = "startseq " + captions + " endseq"
```
- Tokenization
  - Initializes a Tokenizer to convert words into numerical tokens.
  - Trains on cleaned captions, assigning unique indices to words.
  
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
```
- Define Vocabulary and Caption Length
  - vocab_size: Total unique words from the tokenizer plus one for padding.
  - max_caption_length: Length of the longest caption, used for padding.

```python
vocab_size = len(tokenizer.word_index) + 1
max_caption_length = max(len(caption.split()) for caption in captions)
```
- Convert tokens to sequences 
  - Converts each token into a sequence of integers, where each word is replaced by its corresponding token index from the tokenizer.
  
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
- Feature Extraction from Images using ResNet152

| **Model**            | **Best For**                          | **Feature Vector Size** | **Accuracy**     | **Speed**     |
|----------------------|--------------------------------------|------------------|---------------|-------------|
| **DenseNet201**      | Good feature reuse, compact model  | 1920             | 👍 High       | ⚡ Medium   |
| **EfficientNet-B4**  | Best balance of speed & accuracy   | 1792             | 🚀 Higher     | ⚡ Faster   |
| **ConvNeXt-Large**   | State-of-the-art deep features     | 2048+            | 🔥 Very High  | ⏳ Slower   |
| **ResNet152**        | Robust & widely used               | 2048             | 👍 High       | ⚡ Medium   |

This code extracts **deep learning-based features** from images using the **ResNet152** model, a widely used convolutional neural network (CNN) pre-trained on **ImageNet**.

```python
base_model = ResNet152(weights='imagenet', include_top=False, pooling='avg')
feature_model = Model(inputs=base_model.input, outputs=base_model.output)
```

- Preprocesses images using ImageNet norms.
  
```python
img = tf.keras.applications.imagenet_utils.preprocess_input(img)
```


### **📌 Step 2: Encode**  
#### 2.1 Image Features Encoding with CNNs 
- Reduce dimensionality using `Dense(256)` to ensure a compact representation of the image.  
- Convert features to 3D using `Reshape()`
  - `(batch_size, 256) → (batch_size, 1, 256)`
  - This ensures image features align with textual input sequences.
  
```python
input1 = Input(shape=(2048,))
img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)  # Convert to 3D
```

#### 2.2 Text Features Encoding with LSTMs
- Convert words into vector representations using an `Embedding` layer.  
- Transform text into 3D format, which is required for sequential processing in LSTM.

```python
input2 = Input(shape=(max_caption_length,))
text_features = Embedding(vocab_size, embedding_dim, mask_zero=True)(input2)
text_features = LSTM(lstm_units, return_sequences=True)(text_features)
```

#### 2.3 Concatenate Captions and Images to Fit in LSTM Model  
- Two-step merging process:  
  1. Concatenate visual and textual features.
     ```python
     merged = concatenate([img_features_reshaped,sentence_features],axis=1)
     ```
  2. LSTM processes the combined sequence, learning relationships between image content and text.
     ```python
     sentence_features = LSTM(256)(merged)
     ```

### **📌 Step 3: Decode**  

#### 3.1 Use a Dense + LSTM-based Decoder to Predict the Next Word in the Sequence
- LSTM generates the caption word by word, predicting the next word based on the image and previous words.  

#### 3.2 Model Refinement
- Use two `Dense` layers to extract key features and improve accuracy.  
- Apply a `softmax` layer to select the most probable word from the vocabulary.  
- Use Dropout (0.5) to prevent overfitting and improve generalization.  

```python
decoder = Dense(128, activation='relu')(decoder)
decoder = Dense(64, activation='relu')(decoder)
decoder = Dropout(0.5)(decoder) 
output = Dense(vocab_size, activation='softmax')(decoder)
```

### **📌 Step 4: Model Fit** 
- Data Generator Setup: Loads images and captions dynamically in batches, preventing memory overload.

```python
train_generator = CustomDataGenerator(df=train, X_col='image', y_col='caption', batch_size=64, 
                                      directory=image_path, tokenizer=tokenizer, vocab_size=vocab_size, 
                                      max_length=max_caption_length, features=features)

validation_generator = CustomDataGenerator(df=test, X_col='image', y_col='caption', batch_size=64, 
                                           directory=image_path, tokenizer=tokenizer, vocab_size=vocab_size, 
                                           max_length=max_caption_length, features=features)
```
- Callback
  - ModelCheckpoint: Saves the model automatically whenever the validation loss improves, ensuring the best version is retained.
  - EarlyStopping: Stops training early if the validation loss does not improve for a set number of epochs, preventing overfitting.
  - ReduceLROnPlateau: Reduces the learning rate when the validation loss stops improving, helping the model fine-tune its learning.

```python
checkpoint = ModelCheckpoint(model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=1e-8)
```

---

## **📊 3. Model Results & Testing**  
- The graph shows the training loss (blue line) and validation loss (orange line) over multiple epochs.
  - The training loss (blue line) steadily decreases, showing that the model is effectively learning from the data.
  - The validation loss does not immediately increase; instead, it gradually stabilizes around epoch 6-7. This means the model has generalized well up to that point.
![1c3123595ccabb25c4d7fa19bee3c91](https://github.com/user-attachments/assets/38582913-d6c3-447c-86ee-617a57df6309)

- Test
  - "startseq" indicates beginning of the sentence
  - "endseq" indicates end of the sentence
![6a5bc70ed67dbdfaaa373a98647737d](https://github.com/user-attachments/assets/f6fd8f4d-0277-4fae-9bc1-f05f7b7c3807)

- Future Improvement
    - Data Augmentation
      - Augment images with random cropping, flipping, color jittering to improve model robustness.
      - Augment captions using paraphrasing models (e.g., T5, Pegasus) to increase diversity.
    - Better Tokenization
      - Instead of a simple Tokenizer(), consider SentencePiece (Unigram/Lattice model) or Byte-Pair Encoding (BPE) for handling rare words and out-of-vocabulary tokens more efficiently.
