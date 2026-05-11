**Functional Connectivity Cognitive Load Classifier** | Python, scikit-learn, NumPy, Pandas, Seaborn  
Developed a Support Vector Machine (SVM) pipeline to classify human cognitive load states (0-back vs. 2-back working memory) using fMRI functional connectivity data.  
- Processed complex BOLD signal timeseries from the Human Connectome Project (HCP), aggregating data across 360 cortical parcels into intrinsic brain networks.  
- Engineered feature vectors by actively computing and extracting upper-triangular functional correlation matrices for single-network and cross-network analysis.  
- Trained and evaluated a linear SVM classifier with Group Shuffle Split cross-validation (preventing subject data leakage) to robustly predict working memory states.  
- Implemented Multi-dimensional Scaling (MDS) via Euclidean distance on Fisher Z-transformed correlations to mathematically visualize network topology shifts between cognitive tasks.

**Robotic Arm Gesture Control System** | Python, OpenCV, MediaPipe, scikit-learn, Pandas  
Engineered a hybrid computer vision and sensor-fusion platform to explicitly map human hand movements to robotic arm controls.  
- Built a real-time computer vision tracker pipeline using OpenCV and Google MediaPipe to capture and extract continuous 3D coordinate data for 21 hand landmarks.  
- Designed an algorithmic geometric calculator to compute precise phalange joint angles in real time using vector dot products, enabling dynamic robotic actuation.  
- Trained and validated a RandomForest machine learning classifier on multi-axis MPU sensor data with engineered temporal features to accurately predict positional displacements.  
- Processed real-time MPU datastreams via Pandas, optimizing relative coordinate shifts and time-delta differentials for predictive inference.

**Historic Artwork Forgery Detection** | PyTorch, ResNet-50, GANs, Streamlit, Pandas  
Engineered a deep learning platform to authenticate and classify historic artworks using CNNs and generative models.  
- Trained a multitask ResNet-50 computer vision model on a dataset of 45,000+ paintings to accurately predict artist, period, and nationality metadata.  
- Researched and integrated Generative Adversarial Networks (GANs) to synthesize artificial artworks and systematically train a forgery-detection classifier.  
- Built and deployed an interactive Streamlit web application, allowing users to upload artworks for real-time inference and metadata retrieval.  
- Addressed extreme class imbalances across historical periods and artists through comprehensive exploratory data analysis and dataset curation.

**Image Captioning Engine** | Python, TensorFlow, Keras, NumPy, Pillow  
Engineered an encoder-decoder neural network to automatically generate natural language descriptions from raw image inputs.  
- Designed a hybrid CNN-RNN architecture, integrating a pre-trained Xception feature extractor with an LSTM sequence model.  
- Optimized training memory overhead via a `tf.data.Dataset` generator, efficiently processing 40,000+ caption sequences in batches.  
- Mitigated overfitting by applying 50% dropout layers to the 256-dimensional image and text embedding representations.  
- Built an automated CLI inference pipeline merging real-time image preprocessing with iterative word-by-word sequence prediction.

**Automated Receipt OCR Pipeline** | Python, EasyOCR, Pandas, Pillow, Regex  
Developed a computer vision pipeline to extract and digitize structured product and pricing data from raw receipt images.  
- Implemented an EasyOCR text recognition system, parsing bounding box coordinates to geometrically align receipt lines from top to bottom.  
- Engineered custom Regex-based string cleaning and heuristic filtering to automatically eliminate receipt metadata (subtotals, taxes, card details).  
- Built a Pandas data processing module to validate extracted prices and format unstructured OCR text into clean, usable CSV datasets.  
- Optimized data quality via multi-layered validation logic, systematically filtering out entries with extreme alphanumeric noise or insufficient word counts.
