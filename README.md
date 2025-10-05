
## 🩺 Obesity Classification System

### 📘 Overview

The **Obesity Classification System** is a machine learning–based web application that predicts a person’s obesity level based on their lifestyle and physical attributes.
It uses **Python (scikit-learn)** for model training and **Streamlit** for the frontend interface, allowing users to easily check their obesity risk through an interactive web form.

---

### 🎯 Objectives

* Help users understand their obesity risk level.
* Encourage healthy lifestyle changes with personalized feedback.
* Demonstrate an end-to-end data science product — from preprocessing and model training to a deployable software solution.

---

### 🧠 Machine Learning Models Used

We trained and compared multiple classification algorithms:

* **Logistic Regression**
* **Random Forest Classifier**
* **K-Nearest Neighbors (KNN)**

After evaluation, the best-performing model was integrated into the Streamlit web app for real-time predictions.

---

### ⚙️ Tech Stack

**Backend / Model**

* Python 3
* scikit-learn
* pandas, numpy

**Frontend / UI**

* Streamlit

**Version Control**

* Git & GitHub

---

### 🧩 System Architecture

1. **Data Preprocessing**

   * Cleaned and encoded the Obesity dataset.
   * Scaled numerical features for better model performance.
   * Saved the final preprocessed dataset and trained model using `joblib`.

2. **Model Training & Evaluation**

   * Trained Logistic Regression, Random Forest, and KNN models.
   * Evaluated using accuracy and classification reports.
   * Selected the best model for deployment.

3. **Frontend Integration**

   * Built a Streamlit-based web interface.
   * Users can input personal and lifestyle data to get predictions.
   * The app displays the predicted obesity class and relevant feedback areas (e.g., *Low hydration*).


---

### 💻 How to Run the Project

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/FDM_Preprocessing.git
cd FDM_Preprocessing
```

#### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` doesn’t exist yet, just run)*

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

#### 3️⃣ Run the app

```bash
streamlit run app.py
```

#### 4️⃣ Open in browser

Visit: [http://localhost:8501](http://localhost:8501)
You’ll see your Obesity Classification web app running!

---

### 📊 Example Output

After submitting your details, the app shows:

* **Predicted Category:** e.g., *Overweight_Level_I*
* **BMI:** e.g., *24.22*
* **Activity/Day:** e.g., *1.0h*
* **Hydration:** e.g., *2.0L*
* **Attention Areas:** e.g., *Low hydration*

---


### 🚀 Future Improvements

* Deploy on cloud (e.g., Streamlit Cloud or Render).

---





