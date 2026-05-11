**Image Captioning Engine** | Python, TensorFlow, Keras, NumPy, Pillow  
Engineered an encoder-decoder neural network to automatically generate natural language descriptions from raw image inputs.  
- Designed a hybrid CNN-RNN architecture, integrating a pre-trained Xception feature extractor with an LSTM sequence model.  
- Optimized training memory overhead via a `tf.data.Dataset` generator, efficiently processing 40,000+ caption sequences in batches.  
- Mitigated overfitting by applying 50% dropout layers to the 256-dimensional image and text embedding representations.  
- Built an automated CLI inference pipeline merging real-time image preprocessing with iterative word-by-word sequence prediction.
