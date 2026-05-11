**Functional Connectivity Cognitive Load Classifier** | Python, scikit-learn, NumPy, Pandas, Seaborn  
Developed a Support Vector Machine (SVM) pipeline to classify human cognitive load states (0-back vs. 2-back working memory) using fMRI functional connectivity data.  
- Processed complex BOLD signal timeseries from the Human Connectome Project (HCP), aggregating data across 360 cortical parcels into intrinsic brain networks.  
- Engineered feature vectors by actively computing and extracting upper-triangular functional correlation matrices for single-network and cross-network analysis.  
- Trained and evaluated a linear SVM classifier with Group Shuffle Split cross-validation (preventing subject data leakage) to robustly predict working memory states.  
- Implemented Multi-dimensional Scaling (MDS) via Euclidean distance on Fisher Z-transformed correlations to mathematically visualize network topology shifts between cognitive tasks.
