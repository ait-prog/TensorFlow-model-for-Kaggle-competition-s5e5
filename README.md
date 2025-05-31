#  Calorie Prediction using TensorFlow

This notebook builds a neural network model to predict calories burned based on physiological and activity features.

---

##  Datasets Used

- `train.csv` — training set  
- `test.csv` — test set  
- `calories.csv` — additional original data  
- `sample_submission.csv` — submission format

---

##  Features & Engineering

- Dropped `id` column  
- Mapped `Sex`: female → 0, male → 1  
- Created new feature `AgeSex` = `Age` + `Sex`, then label encoded  
- Created interaction features:
  - Multiplication: `feature1_x_feature2`
  - Division: `feature1_div_feature2`
- Added BMI: `Weight / (Height/100)^2`

---

##  Preprocessing

- Applied `StandardScaler` to features  
- Target (`Calories`) transformed with `np.log1p`, inverse with `np.expm1`

---

##  Model

- Sequential neural network:
  - Dense(256) + ReLU + BatchNorm + Dropout(0.3)
  - Dense(128) + ReLU + BatchNorm + Dropout(0.2)
  - Dense(64) + ReLU + BatchNorm
  - Dense(1)
- Optimizer: Adam (`lr=0.001`)
- Loss: Mean Squared Error  
- Metric: Root Mean Squared Error  
- Callbacks:
  - EarlyStopping (patience = 10)
  - ReduceLROnPlateau

---

##  Training

- 5-Fold Cross-Validation (`KFold`)
- Averaged test predictions from all folds

---

##  Output

- Final predictions saved to `submission_tf.csv`

---

##  Tools & Libraries

- Python, NumPy, Pandas  
- TensorFlow / Keras  
- scikit-learn  
- itertools (for feature combinations)

---
