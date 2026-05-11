**Robotic Arm Gesture Control System** | Python, OpenCV, MediaPipe, scikit-learn, Pandas  
Engineered a hybrid computer vision and sensor-fusion platform to explicitly map human hand movements to robotic arm controls.  
- Built a real-time computer vision tracker pipeline using OpenCV and Google MediaPipe to capture and extract continuous 3D coordinate data for 21 hand landmarks.  
- Designed an algorithmic geometric calculator to compute precise phalange joint angles in real time using vector dot products, enabling dynamic robotic actuation.  
- Trained and validated a RandomForest machine learning classifier on multi-axis MPU sensor data with engineered temporal features to accurately predict positional displacements.  
- Processed real-time MPU datastreams via Pandas, optimizing relative coordinate shifts and time-delta differentials for predictive inference.
