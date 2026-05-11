# Internship-Level Interview Questions Based on Your Project Code

This document contains over 100 entry-level, fundamental interview questions tailored directly to the code found in your projects. These questions are designed for an internship interview, focusing on basic ML/DL concepts, code understanding, and standard data science practices.

## 1. Problem Understanding
1. **Q:** In your Image Captioning project, you used both an Xception model and an LSTM. Can you explain in simple terms what each of these models does in the pipeline?
   * **Hint:** Xception extracts features from the image (encoder), and LSTM generates the text sequence (decoder).
   * **Follow-up:** Why is an LSTM better suited for text generation than a standard Artificial Neural Network (Dense layer)?
2. **Q:** For the receipt extraction project, you used EasyOCR and regex rules instead of an end-to-end Machine Learning model. Why might rule-based approaches be useful as a starting point?
   * **Hint:** They are fast to build, require zero training data, and are highly interpretable.
   * **Follow-up:** What is the main disadvantage of relying purely on regex when the receipts come in many different formats?
3. **Q:** In your artwork project, you built a model to predict both 'Period' and 'Nationality' at the same time (Multi-Task Learning). What is the main benefit of doing this instead of training two separate models?
   * **Hint:** It saves memory and compute time because both tasks share the same feature-extraction backbone (ResNet50).
   * **Follow-up:** Do you think predicting Nationality helps the model learn better features for predicting the Period?
4. **Q:** You trained a GAN to generate fake artwork of size 64x64. Why not train it to generate full 1080p high-resolution images right away?
   * **Hint:** High-resolution GANs are extremely difficult and slow to train, and often suffer from severe instability and memory limits.
   * **Follow-up:** How does 64x64 resolution impact the kind of details (like brush strokes) the model can learn?
5. **Q:** In the 10k artist classifier, you chose to limit your data to only the first 10,000 rows (`df.iloc[:10000]`). In a real-world scenario, what factors decide how much data you can train on?
   * **Hint:** Constraints like limited RAM, GPU memory, training time, or experimentation speed.
   * **Follow-up:** If you had enough compute, how would adding 90,000 more images affect the model's performance?

## 2. Data & Preprocessing
6. **Q:** In `model.py`, you add `<start>` and `<end>` tokens to your text captions. Why are these tokens necessary for the LSTM?
   * **Hint:** They tell the model exactly when to begin generating the caption and when the sentence is officially finished.
   * **Follow-up:** What would happen if the model never learned to output the `<end>` token?
7. **Q:** In your receipt pre-processing, you filter out rows that have words like `"total"` or `"tax"`. Can you explain the logic behind this cleaning step?
   * **Hint:** The goal was likely to extract individual product items and their prices, so summary lines are considered noise.
   * **Follow-up:** If a store sells a product literally called "Tax Calculator", how would this script handle it?
8. **Q:** In `train_artist_classifier_10k.py`, you used `stratify=df['artist_encoded']` when splitting into train and test sets. Why is `stratify` important here?
   * **Hint:** It ensures the train and test sets have the same proportion of classes, meaning rare artists won't accidentally be missing from the training set.
   * **Follow-up:** What is the difference between an 80/20 random split and an 80/20 stratified split?
9. **Q:** Looking at `preprocessing.py`, you drop any product name where more than 50% of the characters are numbers. What kind of dirty data were you trying to remove?
   * **Hint:** Barcodes, long product IDs, or serial numbers that OCR accidentally picked up.
   * **Follow-up:** How could you easily test if this rule is dropping real items by mistake?
10. **Q:** Before passing images to ResNet50, you resize them to 224x224. Why is this specific resizing step required?
    * **Hint:** Pre-trained models like ResNet50 expect a specific, fixed input shape because their internal dense layers rely on exact dimensions.
    * **Follow-up:** If an original painting is wide (like a panorama), what happens to the image when you squish it into a 224x224 square?
11. **Q:** In `ext.py`, you sort bounding boxes by their Y-coordinate to read top-to-bottom. Could this cause issues if the receipt has two columns side-by-side?
    * **Hint:** Yes, items in the left and right columns might have the same Y-coordinate and get jumbled together.
    * **Follow-up:** How might you use the X-coordinate to fix this column issue?
12. **Q:** You convert your artwork labels into integers using `LabelEncoder()`. Why do neural networks need string labels (like "Picasso") converted into numbers?
    * **Hint:** Neural networks only perform mathematical operations (addition, multiplication), which require numerical inputs.
    * **Follow-up:** What is the difference between an integer label (e.g., `0`, `1`, `2`) and a one-hot encoded label (e.g., `[1,0,0]`)?

## 3. Model Architecture
13. **Q:** What does `Dropout(0.5)` do in your Image Captioning model, and why is it important?
    * **Hint:** It randomly turns off 50% of the neurons during training. It forces the network to not rely entirely on specific neurons, reducing overfitting.
    * **Follow-up:** Does Dropout stay active when you test or deploy the model?
14. **Q:** You used `ResNet50(weights='imagenet')`. What does it mean to use 'imagenet' weights, and what is Transfer Learning?
    * **Hint:** The model has already been pre-trained on millions of standard images, learning basic shape and edge detection that can be reused on artwork.
    * **Follow-up:** Why is transfer learning often better than training a network from scratch on a small dataset?
15. **Q:** Your image captioning dense layer uses a `relu` activation. What is ReLU, and why is it so common in hidden layers?
    * **Hint:** ReLU (Rectified Linear Unit) outputs the input directly if positive, or zero if negative. It’s fast to compute and helps prevent vanishing gradients.
    * **Follow-up:** What is the "dying ReLU" problem?
16. **Q:** The final layer of your artist classifier uses `activation='softmax'`. What does the softmax function output?
    * **Hint:** It outputs a list of probabilities that sum to 1.0, making it perfect for multi-class classification.
    * **Follow-up:** If you used a `sigmoid` activation instead of `softmax` here, what would happen to the output probabilities?
17. **Q:** In `train_gan.py`, you have a Generator and a Discriminator. Can you explain their basic roles in a competitive format?
    * **Hint:** The Generator tries to create fake data to fool the Discriminator. The Discriminator tries to tell the difference between real and fake data.
    * **Follow-up:** If the Discriminator is significantly stronger than the Generator, what happens to the training?
18. **Q:** Why does your multi-task ResNet50 replace the final fully-connected (fc) layer with `nn.Identity()`?
    * **Hint:** It removes the standard 1000-class ImageNet output so the extracted features can be routed to your custom period and nationality classification heads.
    * **Follow-up:** What is the shape (dimensionality) of the feature vector that ResNet50 produces before the final dense layer?
19. **Q:** In `model.py`, what is the purpose of the `Embedding` layer used for text data?
    * **Hint:** It converts integer word tokens into dense numerical vectors of fixed size (e.g., 256), capturing semantic relationships between words.
    * **Follow-up:** How is an Embedding representation different from a One-Hot encoded word?

## 4. Training Details
20. **Q:** What does the `batch_size` parameter mean in your training scripts?
    * **Hint:** It defines the number of samples the model processes before updating its internal weights (parameters).
    * **Follow-up:** Name one advantage of using a small batch size (e.g., 16 or 32) instead of processing the entire dataset at once.
21. **Q:** In PyTorch, you call `optimizer.zero_grad()` at the start of your training loop. Why is it important to zero out gradients?
    * **Hint:** PyTorch accumulates (adds up) gradients automatically. If you don't zero them, the new gradients will be improperly added to the old ones from the previous step.
    * **Follow-up:** What does `loss.backward()` actually do under the hood?
22. **Q:** In `model.py`, you train for `epochs = 20`. What does one "epoch" represent?
    * **Hint:** One complete pass through the entire training dataset.
    * **Follow-up:** If your training loss is still decreasing rapidly at epoch 20, what should your next step be?
23. **Q:** What is the purpose of an Optimizer like Adam (which you use in multiple scripts)?
    * **Hint:** The optimizer uses the calculated gradients to actually update the weights of the network specifically to minimize the current loss.
    * **Follow-up:** Adam includes the concept of "Momentum". How would you describe momentum simply?
24. **Q:** You use a Learning Rate of `1e-4` in your GAN and PyTorch model. What is a "Learning Rate", and why is choosing the right one important?
    * **Hint:** It controls how big of a step the optimizer takes when updating weights. Too high = the model crashes/diverges. Too small = training takes forever.
    * **Follow-up:** What typically happens to the loss graph if the learning rate is way too large?
25. **Q:** In `main.py`, you extract image features in a simple `for` loop using `tqdm`. How does `tqdm` help a developer during long processes?
    * **Hint:** It displays a visual progress bar and estimates time remaining, helping ensure the script hasn't crashed.
    * **Follow-up:** In Python, is a standard `for` loop across thousands of files utilizing all CPU cores, or just one?
26. **Q:** In `model.py`, why did you use `Model.save('models/model_{i}.h5')` instead of just saving the very last model?
    * **Hint:** In case the model begins to overfit after a certain epoch, you can load an earlier, better-performing version.
    * **Follow-up:** What is the standard Keras callback that stops training automatically when the model stops improving?

## 5. Loss Function & Metrics
27. **Q:** You use `categorical_crossentropy` as your loss function for the artist classifier. Why is this standard for multi-class problems?
    * **Hint:** It measures the distance between the predicted probability distribution and the actual true label, heavily penalizing confident wrong answers.
    * **Follow-up:** What is the difference between `binary_crossentropy` and `categorical_crossentropy`?
28. **Q:** What is a "Loss Function", and why does the ML model need it to train?
    * **Hint:** A mathematical way to measure how "wrong" the model's current predictions are. The optimizer needs this score to know which way to adjust the weights.
    * **Follow-up:** Can we use "Accuracy" directly as a loss function for gradient descent? Why or why not?
29. **Q:** In your PyTorch multi-task model, your loss is `loss1 + loss2`. Does this mean both tasks are treated as equally important during training?
    * **Hint:** Yes, a simple addition gives equal weight.
    * **Follow-up:** If the "Period" loss was generally 10x larger than the "Nationality" loss, how might you balance them in the code?
30. **Q:** The GAN uses `BinaryCrossentropy`. Since it's a generator, why is the loss binary?
    * **Hint:** Because the Discriminator is technically performing a binary classification task: "Real" vs "Fake".
    * **Follow-up:** If the Discriminator's accuracy hits 100% too quickly, why is that actually bad for the Generator?
31. **Q:** In your Receipt Extraction code, there is no AI training loop, just regex. How would you evaluate if your script was performing well?
    * **Hint:** Manually label 100 receipts (ground truth), run the script, and compute the percentage of correctly extracted items.
    * **Follow-up:** What is the difference between calculating "Precision" and "Recall" for extracted lines?

## 6. Overfitting & Regularization
32. **Q:** What is "Overfitting", and how can you tell if your artist classifier is overfitting based on your training and validation accuracy?
    * **Hint:** Overfitting is when the model memorizes the training data but fails on new data. You see it when Training accuracy goes near 100% but Validation accuracy stays low or decreases.
    * **Follow-up:** Name two simple things you can do to reduce overfitting in image models.
33. **Q:** In the 10k artist classifier, you use `val_df = train_test_split(..., test_size=0.2)`. Why is a validation set completely vital to ML model development?
    * **Hint:** It acts as unseen test data during development to ensure the model is actually generalizing and not just memorizing the training set.
    * **Follow-up:** What is a "data leak" between the training and validation sets?
34. **Q:** You used ImageNet weights (`include_top=False`). How does transfer learning implicitly help reduce overfitting when you have a small dataset?
    * **Hint:** Because the base layers are already fully trained on general images, the model needs to learn far fewer new parameters from scratch.
    * **Follow-up:** What does "freezing" layers mean in the context of transfer learning?
35. **Q:** In your pre-processing code, could applying image augmentations (like random flips or rotations) prevent overfitting? Why?
    * **Hint:** Yes. Augmentation artificially increases the size of your dataset and forces the model to recognize patterns from different angles, making it more robust.
    * **Follow-up:** Why might horizontal flipping be a bad augmentation to use when classifying text in images (like receipts)?
36. **Q:** Does simplifying a model architecture (e.g. fewer dense neurons) act as a form of regularization?
    * **Hint:** Yes, limiting model capacity natively prevents it from being complex enough to memorize noise.
    * **Follow-up:** What happens if the model architecture is *too* simple (underfitting)?

## 7. Hyperparameter Choices
37. **Q:** In image captioning, you set the LSTM and Dense layers to `256` units. What would happen if you changed this to `2` units?
    * **Hint:** The model would lack the "brain capacity" to store the complexity of paragraph structures, resulting in terrible performance.
    * **Follow-up:** What would be the downside of setting that number to `100,000`?
38. **Q:** You chose to train the GAN for `epochs = 4`. Considering GANs are notoriously hard to train, is 4 epochs generally enough?
    * **Hint:** Usually, no. GANs typically require hundreds or thousands of epochs to begin generating anything recognizable.
    * **Follow-up:** What is an easy visual way to track if a GAN is improving every epoch?
39. **Q:** `batch_size` in the PyTorch code is 16. If your GPU crashes with an "Out of Memory" error, what is the easiest hyperparameter to adjust?
    * **Hint:** Lowering the `batch_size` directly reduces the memory footprint required to hold the tensors.
    * **Follow-up:** If you lower the batch size to 2, will training take more time or less time to complete an epoch?
40. **Q:** You hardcoded `conf > 0.3` in the receipt text extraction. What happens to the output if you set this to `0.95`?
    * **Hint:** The script will filter out a massive amount of text, throwing away a lot of valid receipt items that the OCR wasn't 100% confident in.
    * **Follow-up:** Does a high threshold favor Precision or Recall?

## 8. Code Design & Basic Engineering
41. **Q:** In your PyTorch Dataset `__getitem__`, you have an `except Exception:` block that catches errors when an image fails to open. Why is `try-except` important in data pipelines?
    * **Hint:** It prevents one single corrupted `.jpg` file from crashing a training job that might have been running for 5 hours.
    * **Follow-up:** Is it a good practice to use a bare `except Exception:` without printing the error message? Why or why not?
42. **Q:** In `main.py`, you use `os.path.join(directory, img)`. Why is this preferred over just typing `directory + "/" + img`?
    * **Hint:** `os.path.join` natively handles file path differences across various operating systems (Windows uses `\`, Mac/Linux use `/`).
    * **Follow-up:** In Python 3, what is the modern library commonly used for path management that replaces `os.path`?
43. **Q:** Your image captioning code saves features to `features.p` using the `pickle` module. What is the fundamental purpose of Python Pickling?
    * **Hint:** Pickling serializes (saves) a Python object (like a dictionary or list) directly to disk so it can be loaded instantly later without recalculating.
    * **Follow-up:** Why is sharing raw `.p` (pickle) files with strangers online considered a security risk?
44. **Q:** Your Pandas script uses `df.to_csv("receipts_final_cleaned.csv", index=False)`. What does `index=False` do?
    * **Hint:** It prevents Pandas from writing the arbitrary row numbers (0, 1, 2) as the very first column in the CSV file.
    * **Follow-up:** How do you view the first 5 rows of a Pandas dataframe?
45. **Q:** Most of your scripts include `if __name__ == "__main__":`. What does this line do in Python?
    * **Hint:** It ensures the code inside the block only runs if the script is executed directly, not if it is imported as a module by another script.
    * **Follow-up:** If you didn't have this, and imported your script, what would automatically execute?
46. **Q:** Why do we import libraries like `numpy as np` and `pandas as pd`?
    * **Hint:** It is standard community convention to use these shorthands to keep code clean and readable.
    * **Follow-up:** What is Numpy primarily used for in Machine Learning pipelines?

## 9. Conceptual "What If" Scenarios
47. **Q:** **What if** you were asked to deploy the Receipt Extraction code (`ext.py`) as a web API used by thousands of users? What is a basic framework you could use?
    * **Hint:** Mention Python web frameworks like Flask or FastAPI.
    * **Follow-up:** What data format would the API most likely return? (Hint: JSON).
48. **Q:** **What if** your model was getting 95% accuracy on predicting artwork during the internship, but when you deployed it to users, it failed constantly? What might be the issue?
    * **Hint:** Domain mismatch / Data drift. Maybe the users are uploading blurry phone pictures, while your training data consisted of perfectly cropped museum scans.
    * **Follow-up:** How could you fix this mismatch using your training data?
49. **Q:** **What if** the dataset of old artwork had 5,000 paintings from the "Renaissance" but only 10 from the "Modern" period. How would this imbalance affect the model?
    * **Hint:** The model would become heavily biased toward guessing "Renaissance" for almost everything, and would perform terribly on "Modern" art.
    * **Follow-up:** What is "oversampling" in this context?
50. **Q:** **What if** the business asked you to explain exactly *why* your PyTorch ResNet model predicted a painting was from a specific period. With deep learning, is this easy to do?
    * **Hint:** No. Deep learning models are generally considered "black boxes", unlike simpler models like Decision Trees.
    * **Follow-up:** Have you heard of tools (like CAM or Grad-CAM) that can highlight which pixels the CNN was looking at?
51. **Q:** **What if** your training data contained no captions longer than 5 words, but in the real world, users expect full descriptive paragraphs?
    * **Hint:** The model can only learn what it has seen. It will physically be incapable of generating realistic long paragraphs.
    * **Follow-up:** Does having extremely long sentences (pad sequences) drastically increase training time?
52. **Q:** **What if** your manager asked you to cut down the size of the 100MB model file so it can fit on a mobile phone app. Do you know any ML concepts that deal with this?
    * **Hint:** Model compression, smaller architectures (MobileNet instead of ResNet), or Quantization (changing math float precision).
    * **Follow-up:** How does shrinking model size usually affect accuracy?
