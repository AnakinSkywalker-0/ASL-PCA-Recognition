# ASL Real-Time Recognition — PCA Subspace Method
### Linear Algebra & AIML Project

This project implements a real-time American Sign Language (ASL) recognition system using **Principal Component Analysis (PCA)**. Instead of using a deep learning black-box, it relies on **Eigendecomposition** to create mathematical subspaces for various hand gestures.




## 📂 Project Structure
```text
D:/AI/Hand_symbol/
├── word_dataset/         # Reorganized images (hello, thanks, no, etc.)
├── hand_landmarker.task  # MediaPipe pre-trained model
├── gestures.csv          # Extracted 42D coordinates
├── pca_model.npz         # Saved Mean vectors and Eigenvectors
├── extract.py      # Landmark extraction script
├── train_pca.py    # The core Linear Algebra training script
├── gesture_ui.py         # Tkinter-based live recognition interface
└── evaluate.py     # Accuracy reports and visualizations
|-reorganize_data.py    # Cleaning the dataset 
```




## 🧠 The Linear Algebra Explained
Each hand gesture is treated as a vector $v$ in a 42-dimensional space ($\mathbb{R}^{42}$), representing the $(x, y)$ coordinates of 21 hand landmarks.

### 1. Covariance & Eigendecomposition
For each gesture class (e.g., "Hello"), we calculate the **Covariance Matrix** $C$:
$$C = \frac{(X - \mu)^T(X - \mu)}{n - 1}$$
where $X$ is the matrix of samples and $\mu$ is the mean gesture shape. We then find the **Eigenvectors** ($V$) and **Eigenvalues** ($\Lambda$) such that $CV = V\Lambda$.



### 2. Projection and Reconstruction Error
Recognition is performed by projecting a live hand vector $v_{live}$ onto the subspace defined by the top-$k$ eigenvectors ($U$) of each class:
$$v_{proj} = U U^T (v_{live} - \mu)$$
The system calculates the **Euclidean distance** (reconstruction error) between the original vector and its projection:
$$d = \|(v_{live} - \mu) - v_{proj}\|$$
The class with the **minimum distance** is identified as the predicted meaning.






### 📥 Data Acquisition & Pre-processing

The model is trained using the [American Sign Language Image Dataset](https://www.kaggle.com/datasets/shubhambhardwaj01/american-sign-language-image-dataset). 

Because the raw dataset is distributed with a non-standard folder structure (e.g., separate `train` and `test` directories with prefixed filenames), a pre-processing step is required to prepare the data for Feature Extraction.

#### 1. Download and Rename
* Download the dataset from Kaggle.
* Rename the root folder of the downloaded images to `word_dataset` and place it in the project directory.

#### 2. Automated Reorganization
Run the provided `reorganize_data.py` script. This script performs the following:
* **Parses Filenames**: Identifies labels from filenames (e.g., `bathroom.a06d.jpg` $\rightarrow$ `bathroom`).
* **Restructures Folders**: Creates individual subdirectories for each gesture (e.g., `./word_dataset/hello/`).
* **Standardizes Layout**: Moves all images into their respective labeled folders so `extract.py` can correctly assign classes to the 42D vectors.

```bash
python reorganize_data.py
```
OR 
Simply just Download the the given word_dataset.zip


### Why this matters for the PCA Model
In our Linear Algebra implementation, the `extract.py` script treats each subfolder name as a unique **Class Label**. By organizing the images into these folders, we ensure that the **Covariance Matrix** computed in the training phase represents the mathematical variance of a specific gesture. Without this reorganization, the model would be unable to distinguish between different "Eigen-gestures".


### Step 1: Extract Coordinates
Run the extractor to process your reorganized `word_dataset`:
```bash
python extract.py --dataset ./word_dataset --out gestures.csv
```

### Step 2: Train the Model
Generate the PCA subspaces. We keep the top 10 principal components:
```bash
python train_pca.py --csv gestures.csv --out pca_model.npz --k 10 --eval
```

### Step 3: Launch the UI
Start the live recognition interface to access the system camera:
```bash
python gesture_ui.py
```

### Step 4: Visualize Results
Generate plots for your report, including **Eigen-gestures** (the "ghost hands" representing variance) and the **PCA Scatter Plot**:
```bash
python evaluate.py --csv gestures.csv --model pca_model.npz
---

---
### **Team Division: ASL PCA Project**

#### **Person 01: The Data Engineer**
* **Focus**: Data Acquisition & Pre-processing.
* **Key Tasks**:
    * **Manage Dataset**: Ensure the `word_dataset/` folder is correctly structured with labeled subfolders.
    * **Feature Extraction**: Run and maintain `extract.py` to use MediaPipe for landmark detection.
    * **Vector Normalization**: Ensure the 42-dimensional vectors ($x,y$ coordinates) are correctly centered on the wrist to maintain position independence.
* **Deliverable**: `gestures.csv` (The raw numerical foundation of the project).

#### **Person 02: The Math Core (Linear Algebraist)**
* **Focus**: Mathematical Modeling & Training.
* **Key Tasks**:
    * **Subspace Construction**: Run `train_pca.py` to calculate the **Mean Vector** ($\mu$) and **Covariance Matrix** ($C$) for every gesture class.
    * **Eigendecomposition**: Perform the spectral decomposition ($CV = V\Lambda$) to extract the top 10 **Principal Components** ($k=10$).
    * **Model Packaging**: Export the basis vectors and means into a compressed format for the live system.
* **Deliverable**: `pca_model.npz` (The mathematical "brain" of the system).

#### **Person 03: The Systems Integrator**
* **Focus**: Real-time Implementation & UI.
* **Key Tasks**:
    * **Live Projection**: Implement the logic in `gesture_ui.py` that projects live webcam vectors onto the saved subspaces: $v_{proj} = UU^T(v_{live} - \mu)$.
    * **Classifier Logic**: Calculate the **Euclidean distance** (reconstruction error) to determine the closest match.
    * **User Interface**: Optimize the Tkinter window to display the camera feed and predicted meanings smoothly.
* **Deliverable**: `gesture_ui.py` (The functional live demo).

#### **Person 04: The Analyst & Presenter**
* **Focus**: Evaluation, Visualization, and Documentation.
* **Key Tasks**:
    * **Model Validation**: Use `evaluate.py` to generate the **Confusion Matrix** and calculate final accuracy scores.
    * **Visual Proof**: Plot the **Eigen-gestures** (ghost hands) and the **PCA Scatter Plot** to show how the math clusters different gestures.
    * **Project Report**: Write the final explanation of why PCA beats "black-box" models for this Linear Algebra course.
* **Deliverable**: `evaluate.py` outputs, Slide Deck, and Final Report.

---

### **Workload Handoff Flow**
| From | To | Shared Asset |
| :--- | :--- | :--- |
| **Person 01** | **Person 02** | `gestures.csv` |
| **Person 02** | **Person 03 & 04** | `pca_model.npz` |
| **Person 03** | **Whole Team** | Live Demo |
| **Person 04** | **Professor** | Final Presentation |
