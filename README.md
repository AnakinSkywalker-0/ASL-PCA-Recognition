ASL Real-Time Recognition — PCA Subspace Method
Linear Algebra & AIML Project
This project implements a real-time American Sign Language (ASL) recognition system using Principal Component Analysis (PCA). Instead of a deep learning black-box, it uses Eigendecomposition to build mathematical subspaces for each hand gesture, then classifies live webcam input by measuring reconstruction error.
---
Project Structure
```
Hand\_symbol/
├── word\_dataset/          # Images organized by gesture label (one subfolder per gesture)
├── word\_dataset.zip       # Pre-packaged dataset (use this instead of Kaggle)
├── hand\_landmarker.task   # MediaPipe model — auto-downloaded on first run
├── gestures.csv           # Extracted 42D landmark vectors (output of extract.py)
├── pca\_model.npz          # Trained PCA model — mean vectors + eigenvectors (output of train\_pca.py)
├── reorganize\_data.py     # Pre-processing script for raw Kaggle dataset
├── extract.py             # Step 1: Extract hand landmarks → gestures.csv
├── train\_pca.py           # Step 2: Build PCA subspaces → pca\_model.npz
├── gesture\_ui.py          # Step 3: Live webcam recognition UI
└── evaluate.py            # Step 4: Accuracy report + visualizations
```
---
The Math (How It Works)
Each hand gesture is represented as a vector v in a 42-dimensional space (ℝ⁴²), made up of the (x, y) coordinates of 21 hand landmarks detected by MediaPipe.
1. Covariance Matrix & Eigendecomposition
For each gesture class, a Covariance Matrix C is computed from all training samples:
$$C = \frac{(X - \mu)^T(X - \mu)}{n - 1}$$
where X is the matrix of samples and μ is the mean gesture shape. Eigendecomposition gives us the Eigenvectors (V) and Eigenvalues (Λ):
$$CV = V\Lambda$$
The top-k eigenvectors form the gesture's PCA subspace — they capture the most variation in that gesture class.
2. Recognition via Reconstruction Error
A live hand vector v_live is projected onto each class's subspace:
$$v_{proj} = UU^T(v_{live} - \mu)$$
The Euclidean distance (reconstruction error) is calculated:
$$d = |(v_{live} - \mu) - v_{proj}|$$
The class with the minimum distance is the predicted gesture. A confidence threshold filters out low-quality predictions.
---
Setup
Prerequisites
```bash
pip install mediapipe opencv-python numpy pandas matplotlib Pillow scikit-learn
```
Getting the Dataset
Option A — Use the included zip (recommended):
Simply unzip `word\_dataset.zip` in the project folder:
```bash
# Windows (cmd)
tar -xf word\_dataset.zip

# Or right-click the zip → Extract Here
```
Option B — Download from Kaggle:
Download the American Sign Language Image Dataset from Kaggle.
Rename the root folder to `word\_dataset` and place it in the project directory.
Run the reorganization script to restructure it:
```bash
python reorganize\_data.py
```
This parses filenames (e.g., `hello.a06d.jpg` → label `hello`) and moves images into labeled subfolders that `extract.py` can read.
> \*\*Important:\*\* The project folder must be the working directory when running all scripts. Do not run them from a different folder or the file paths will break.
---
Running the Project (Step by Step)
Step 0 — Navigate to the project folder
Open Command Prompt and `cd` into the project directory:
```bash
cd path\\to\\Hand\_symbol
```
All subsequent commands must be run from this folder.
---
Step 1 — Extract Hand Landmarks
Processes every image in `word\_dataset/`, detects hand landmarks using MediaPipe, and saves the normalized 42D vectors to a CSV file.
```bash
python extract.py --dataset ./word\_dataset --out gestures.csv
```
On first run, this auto-downloads `hand\_landmarker.task` (~25 MB) from Google's servers.
Output: `gestures.csv`
Images where no hand is detected are skipped and reported.
---
Step 2 — Train the PCA Model
Reads `gestures.csv`, computes the covariance matrix and eigenvectors for each gesture class, and saves the model.
```bash
python train\_pca.py --csv gestures.csv --out pca\_model.npz --k 10 --eval
```
Flag	Description
`--k 10`	Number of principal components to keep per gesture (default: 10)
`--eval`	Run an accuracy check on the training data after building the model
`--no-plot`	Skip generating the PCA scatter plot
Output: `pca\_model.npz` and `pca\_model\_pca\_scatter.png`
---
Step 3 — Launch the Live Recognition UI
Opens a webcam window with real-time gesture detection and a side panel showing all class distances.
```bash
python gesture\_ui.py
```
Requires `pca\_model.npz` (from Step 2) and `hand\_landmarker.task` (from Step 1) to be in the same folder.
If your webcam doesn't open, edit `gesture\_ui.py` and change `CAMERA\_ID = 0` to `CAMERA\_ID = 1`.
Use the threshold slider in the UI to adjust confidence sensitivity. Lower = stricter.
Press Save Screenshot to save the current frame.
---
Step 4 — Evaluate and Visualize
Splits `gestures.csv` into train/test sets, runs the classifier, and generates output plots.
```bash
python evaluate.py --csv gestures.csv --model pca\_model.npz
```
Outputs:
`confusion\_matrix.png` — per-class accuracy breakdown
`eigengestures.png` — "ghost hand" visualizations of each class's eigenvectors
`pca\_scatter.png` — 2D projection of all gesture clusters
---
Troubleshooting
Error	Fix
`FileNotFoundError: hand\_landmarker.task`	Run `extract.py` first — it auto-downloads the model. Or check that `hand\_landmarker.task` is in the same folder as the scripts.
`PCA model not found: pca\_model.npz`	Run `train\_pca.py` first (Step 2). Make sure you're running from the correct folder.
`NameError: mp\_python`	Old version of `gesture\_ui.py`. Add `from mediapipe.tasks import python as mp\_python` near the top imports.
Webcam doesn't open	Change `CAMERA\_ID = 0` to `CAMERA\_ID = 1` in `gesture\_ui.py`.
Many images skipped in Step 1	The dataset images may already have landmarks drawn on them. Try lowering `min\_hand\_detection\_confidence` in `extract.py`.
Low accuracy	Increase dataset size per class or try `--k 15` or `--k 20` in `train\_pca.py`.
---
Team Division
Person 01 — Data Engineer
Responsible for: Data preparation and feature extraction.
Tasks:
Obtain and set up the dataset (either from `word\_dataset.zip` or Kaggle).
If using Kaggle: run `reorganize\_data.py` to restructure the folder layout.
Run `extract.py` to process all images through MediaPipe and extract 42D hand landmark vectors.
Verify `gestures.csv` is correct — check that all gesture labels are present and no class has too few samples.
Understand the normalization logic in `extract.py`: landmarks are centered on the wrist (landmark 0) and scaled by the wrist-to-middle-finger distance. This makes the model position- and scale-independent.
Deliverable: `gestures.csv` — the numerical foundation the entire model is built on.
---
Person 02 — Math Core (Linear Algebraist)
Responsible for: The PCA model and the underlying mathematics.
Tasks:
Run `train\_pca.py` to build the PCA subspaces.
Understand and be able to explain the full math pipeline:
Why we center the data: $X_c = X - \mu$
How the Covariance Matrix is computed: $C = X_c^T X_c / (n-1)$
What eigendecomposition gives us and why `np.linalg.eigh` is used (symmetric matrix, numerically stable)
Why we keep only the top-k eigenvectors (they capture the most variance)
How the variance explained percentage is computed
Experiment with different values of `--k` and observe the effect on accuracy.
Generate the PCA scatter plot and explain what it shows.
Deliverable: `pca\_model.npz` — the trained model containing mean vectors and eigenvectors for each gesture class.
---
Person 03 — Systems Integrator
Responsible for: The real-time recognition UI.
Tasks:
Run `gesture\_ui.py` and verify it works end-to-end with the webcam.
Understand the live classification pipeline in `gesture\_ui.py`:
Each webcam frame → MediaPipe detects landmarks → normalized 42D vector → projected onto each class subspace → minimum reconstruction error → predicted label
The confidence threshold (`THRESHOLD = 0.40`) and how adjusting it affects predictions
Understand the threading architecture: the camera loop runs in a background thread; the UI updates on the main thread every 30ms.
Be able to demo the live system to the class — test all gesture classes and know which ones are more/less reliable.
Fix any hardware-specific issues (camera ID, resolution, lighting).
Deliverable: A working live demo using `gesture\_ui.py`.
---
Person 04 — Analyst & Presenter
Responsible for: Evaluation, visualization, and the final report/presentation.
Tasks:
Run `evaluate.py` to generate all output plots.
Interpret the confusion matrix: which gestures get confused with each other and why.
Interpret the eigengestures plot: what do the "ghost hands" represent mathematically? (They are directions of maximum variance in the gesture's shape space.)
Interpret the PCA scatter plot: are the gesture clusters well-separated? What does overlap mean for classification?
Write the project report explaining:
Why PCA was chosen over deep learning for this course
The full math pipeline end-to-end
Accuracy results and limitations
Prepare the final presentation slides.
Deliverable: `confusion\_matrix.png`, `eigengestures.png`, `pca\_scatter.png`, slide deck, and final report.
---
Handoff Flow
From	To	Asset
Person 01	Person 02	`gestures.csv`
Person 02	Persons 03 & 04	`pca\_model.npz`
Person 03	Whole team	Live demo
Person 04	Professor	Final presentation + report
---
Dependencies
Package	Used in
`mediapipe`	`extract.py`, `gesture\_ui.py`
`opencv-python`	`extract.py`, `gesture\_ui.py`
`numpy`	all scripts
`pandas`	`train\_pca.py`, `evaluate.py`
`matplotlib`	`train\_pca.py`, `evaluate.py`
`Pillow`	`gesture\_ui.py`
`scikit-learn`	`evaluate.py`
