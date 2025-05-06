# ⚡ Energy Predictor (CS643 Programming Assignment 2)

**Author**: Hrishikesh Akkala
**Course**: CS643 - Cloud Computing  
**Instructor**: Borcea Christian 
**Assignment**: Programming Assignment 2  
**Semester**: Spring 2025

---

## 🧠 Project Overview

This project builds a machine learning pipeline using Apache Spark to predict **energy consumption** of buildings based on various features such as:
- Building type
- Square footage
- Occupancy levels
- Appliance usage
- Temperature
- Day of the week

The final model is trained using a **Gradient Boosted Tree Regressor** and evaluated using **RMSE**. The prediction component is containerized with **Docker** to ensure easy deployment.

---

## 🛠️ Technologies Used

- **Apache Spark (PySpark)**
- **GBTRegressor (MLlib)**
- **CrossValidator for hyperparameter tuning**
- **Docker** (for model deployment)
- **Python 3.10**

---

## 🧪 Dataset

Two datasets were used:
- `TrainingDataset.csv`: Used for model training and hyperparameter tuning
- `ValidationDataset.csv`: Used to evaluate model performance

## 🚀 Project Structure

```

energy-predictor/
│
├── dataset/
│   ├── TrainingDataset.csv
│   └── ValidationDataset.csv
│
│
├── training.py             # Builds, tunes, and saves the ML pipeline
├── prediction.py           # Loads model and predicts from a test CSV
├── requirements.txt        # Required Python libraries
├── Dockerfile              # Docker container for running prediction
└── README.md               # You're here

````

---

## ⚙️ How to Run

### 🧑‍💻 1. Install Dependencies

```bash
pip install -r requirements.txt
````

### 🏋️ 2. Train the Model

```bash
python training.py
```

This will:

* Train the model using 3-fold cross-validation
* Save the best pipeline model to `model/energy_model`
* Print RMSE on validation data

### 📈 3. Run Predictions

```bash
python prediction.py dataset/ValidationDataset.csv
```

This will:

* Load the trained model
* Generate predictions
* Print RMSE and display the first 10 predicted values

---

## 🐳 Docker Deployment

To run the prediction component using Docker:

### ✅ 1. Build Docker Image

```bash
docker build -t energy-predictor .
```

### ✅ 2. Run Prediction in Container

```bash
docker run -v $(pwd)/dataset:/app/dataset energy-predictor dataset/ValidationDataset.csv
```

---

## 📊 Sample Output

```
✅ Final RMSE on validation set: 3.87
+-------------------+------------------+
|Energy Consumption |       prediction |
+-------------------+------------------+
|  15.2             |        14.89     |
|  12.6             |        12.77     |
|  ...              |        ...       |
+-------------------+------------------+
```

---

## ✍️ Acknowledgements

Special thanks to the NJIT and the Professor for guiding the course.

---

## 📌 Notes

* The model uses Spark’s `GBTRegressor` with hyperparameter tuning via `ParamGridBuilder`
* One-hot encoding is used for categorical features
* Feature vectors are scaled using `StandardScaler`
* Spark 3.5 and Java 8/11 are required for compatibility
