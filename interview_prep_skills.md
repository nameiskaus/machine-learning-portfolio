# Machine Learning & AI Portfolio: Master Skills List

To confidently discuss and defend your portfolio in technical interviews, focus on mastering the theoretical concepts and practical implementations listed below.

## 1. Deep Learning & Neural Network Architectures
*   **Convolutional Neural Networks (CNNs):**
    *   Understand convolutions, pooling layers, strides, and padding.
    *   *Xception Architecture:* Depthwise separable convolutions.
    *   *ResNet-50 Architecture:* Residual blocks and skip connections. Why do they solve the vanishing gradient problem?
*   **Recurrent Neural Networks (RNNs) & LSTMs:**
    *   The vanishing gradient problem in vanilla RNNs.
    *   LSTM cell architecture: Forget gates, input gates, output gates, and cell states.
    *   Encoder-Decoder architectures (used in Image Captioning).
*   **Generative Adversarial Networks (GANs):**
    *   Architecture: Generator vs. Discriminator.
    *   Loss functions (Min-Max game, Zero-Sum game).
    *   Common issues: Mode collapse, non-convergence.
*   **Transfer Learning & Fine-Tuning:**
    *   Using pre-trained weights (e.g., ImageNet).
    *   Freezing layers vs. Fine-tuning end-to-end.

## 2. Computer Vision (CV) Tools & Pipelines
*   **OpenCV & Image Processing:**
    *   Color space conversions (e.g., BGR to RGB).
    *   Image resizing, thresholding, and geometric transformations.
*   **MediaPipe:**
    *   Hand landmark detection (21 3D coordinates/nodes).
*   **Optical Character Recognition (OCR):**
    *   How EasyOCR works under the hood.
    *   Handling bounding boxes (`[x_min, y_min, x_max, y_max]`) to geometrically align text.

## 3. Traditional Machine Learning Models
*   **Support Vector Machines (SVMs):**
    *   Concept of the hyperplane, margin, and support vectors.
    *   Linear vs. Non-linear kernels (The Kernel Trick).
*   **Random Forests & Decision Trees:**
    *   Ensemble learning concepts: Bagging (Bootstrap Aggregating) vs. Boosting.
    *   How Random Forests prevent overfitting.
    *   Feature importance extraction.
*   **Dimensionality Reduction:**
    *   Multi-Dimensional Scaling (MDS) vs. PCA or t-SNE.
    *   When and why to use distance/dissimilarity matrices.

## 4. Data Preprocessing & Validation
*   **Handling Unbalanced Data:**
    *   Techniques you could use/have used (e.g., SMOTE, class weights, or GANs for synthetic data).
*   **Cross-Validation Strategies:**
    *   K-Fold vs. Stratified K-Fold.
    *   **Group Shuffle Split:** Why it's critical for medical/fMRI data to prevent data leakage (ensuring the same subject's data doesn't appear in both the train and test sets).
*   **Feature Engineering:**
    *   Timeseries manipulation: Creating time-deltas, relative coordinate shifts (used in MPU sensor data).
    *   Text cleaning: Using **Regex (Regular Expressions)** to filter out noise, extract prices, and standardize product names.

## 5. Python Ecosystem & Frameworks
*   **TensorFlow / Keras vs. PyTorch:**
    *   Differences in philosophy (Dynamic vs. Static computation graphs, eager execution).
    *   When to use `tf.data.Dataset` generators (lazy loading, optimized batching for memory efficiency).
*   **Pandas & NumPy:**
    *   Vectorized operations for speed over `for` loops.
    *   Aggregations, merging datasets, and advanced indexing.
*   **Deployment:**
    *   **Streamlit:** Building interactive ML web applications quickly.

## 6. Essential Math & Theory Concepts
*   **Linear Algebra:**
    *   Dot products, matrix multiplication, vector normalization, and calculating joint angles dynamically.
*   **Statistics:**
    *   **Fisher Z-transformations:** Why convert Pearson correlation coefficients before taking averages or Euclidean distances?
    *   Euclidean distance vs. Cosine Similarity.
