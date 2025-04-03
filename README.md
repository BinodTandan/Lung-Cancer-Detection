# 🫁 Lung Cancer Detection using Machine Learning

This project uses machine learning models to predict the likelihood of lung cancer based on patient features such as age, smoking habits, and symptoms. It includes data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and evaluation.

---

## 🔍 Problem Statement

Early detection of lung cancer is crucial for effective treatment. This project builds a predictive model using clinical and lifestyle data to assist in risk assessment and early detection.

---

## 🧠 Models Used
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost
- Multi Layer Perceptron (MLP)
- Naivee Bayes
- GridSearchCV for hyperparameter tuning

---

## 🧰 Tech Stack

- **Language**: Python
- **Libraries**: Scikit-learn, Pandas, Seaborn, Matplotlib
- **Environment**: Jupyter Notebook, Virtualenv
- **Tools**: VS Code

---

## 📂 Project Structure

Lung Cancer Detection/ │ ├── data/ # Dataset (from Kaggle) ├── notebooks/ # Jupyter Notebooks ├── outputs/ # Results, plots, metrics ├── README.md # This file ├── requirements.txt # Dependencies └── .gitignore # Git exclusions

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/BinodTandan/lung-cancer-detection.git
cd lung-cancer-detection

# Set up virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook notebooks/lung_cancer_classification.ipynb


## 📈 Results    

   Model   CV Loss  Training Loss  Validation Loss  Test Loss  \
0        LOG_REG  0.281437       0.385758         0.239343   0.165154   
1             NB       NaN       0.636408         0.848299   0.454534   
2        XGBOOST  0.240534       0.161734         0.294936   0.289836   
3  MOST_FREQUENT       NaN       4.839194         4.701346   3.067545   
4            MLP  0.268071       0.243572         0.317835   0.238337   
5            SVM  0.274922       0.138538         0.321285   0.264412   
6  RANDOM_FOREST  0.781807       0.087745         1.036414   1.088788   

   Test Accuracy  Precision    Recall  F1 Score  
0       0.936170   0.944174  0.936170  0.939341  
1       0.893617   0.928495  0.893617  0.906201  
2       0.893617   0.928495  0.893617  0.906201  
3       0.914894   0.837030  0.914894  0.874232  
4       0.893617   0.928495  0.893617  0.906201  
5       0.893617   0.928495  0.893617  0.906201  
6       0.872340   0.923350  0.872340  0.890503 

- Added ROC curve, confusion matrix, and feature importance charts.
- Best model: Logistic Regression with highest accuracy.

## 📌 Future Improvements

- Add deep learning model (e.g., MLP or CNN with clinical image data)
- Integrate SHAP or LIME for model interpretability
- Deploy with Streamlit or Flask for live inference
- Create an API endpoint using FastAPI

## 👨‍💻 Author

Binod Tandan




