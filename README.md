# **Iris Classification: A Deployment Learning Project** 

## **📌 Project Overview**
This project is a **toy Machine Learning model** created for the purpose of understanding **deployment workflows, FastAPI, Streamlit, and CI/CD pipelines**. The focus is not on building the best ML model but on learning how to deploy and integrate various technologies. It includes:
- **Exploratory Data Analysis (EDA)**
- **Model Training & Evaluation**
- **Hyperparameter Tuning**
- **API Deployment using FastAPI**
- **Frontend UI using Streamlit**
- **CI/CD & Deployment on Render & Streamlit Cloud**

## **📊 Dataset & Problem Statement**
- **Dataset:** The **Iris dataset** contains 150 samples of iris flowers with four features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
  - Label: One of three classes (Setosa, Versicolor, Virginica)
- **Goal:** Train a model to classify iris flowers based on their measurements.

## **🛠 Tech Stack**
| Component | Technology Used |
|-----------|----------------|
| **Programming Language** | Python 3.10 |
| **Libraries** | Scikit-learn, Pandas, NumPy, Joblib |
| **Model** | Decision Tree |
| **Backend API** | FastAPI |
| **Frontend UI** | Streamlit |
| **Deployment** | Render (FastAPI) & Streamlit Cloud |
| **Version Control** | Git & GitHub |
| **CI/CD** | Render Auto-deploy & Streamlit Cloud Git Sync |

## **📂 Project Structure**
```
📁 decision_tree_project
│── src/
│   ├── fastapi_app.py  # FastAPI backend
│   ├── streamlit_app.py # Streamlit UI
│── models/
│   ├── best_decision_tree.pkl  # Trained model
│   ├── scaler.pkl  # Data scaler
│── requirements.txt  # Dependencies
│── README.md  # Documentation
```

## **🔄 Workflow**
1. **Data Preprocessing & EDA**  📊
2. **Model Training & Hyperparameter Tuning** 🏋️
3. **Saving the Best Model** 🎯
4. **Building an API with FastAPI** 🚀
5. **Creating a UI with Streamlit** 🎨
6. **Deploying FastAPI on Render** 🌍
7. **Deploying Streamlit on Streamlit Cloud** ☁
8. **CI/CD for Auto Deployment** 🔄

## **🚀 Running the Project Locally**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/decision_tree_project.git
cd decision_tree_project
```
### **2️⃣ Create a Virtual Environment & Install Dependencies**
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### **3️⃣ Run FastAPI Backend**
```sh
uvicorn src.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```
- Open **FastAPI Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### **4️⃣ Run Streamlit UI**
```sh
streamlit run src/streamlit_app.py
```
- Open **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

## **🌍 Live Deployment Links**
- **FastAPI Backend:** [https://toy-dt-project.onrender.com](https://toy-dt-project.onrender.com)
- **Streamlit Frontend:** [https://your-streamlit-app.streamlit.app](https://your-streamlit-app.streamlit.app)

## **📌 Challenges & Learnings**
✅ **Challenges Faced:**
- Debugging FastAPI errors (e.g., invalid predictions)
- Handling relative paths for model loading
- Connecting Streamlit UI with FastAPI backend
- CI/CD automation for seamless deployment

✅ **Key Learnings:**
- Deploying machine learning models with **FastAPI & Streamlit**
- Setting up **CI/CD pipelines with Render & Streamlit Cloud**
- Importance of **logging & error handling** for debugging

## **🚀 Next Steps**
- Implement **Docker** for containerized deployment 🐳
- Add **logging & monitoring** for better debugging 📊
- Experiment with **Random Forest & XGBoost** for better accuracy 🌲
- Improve **frontend UI styling** for a better user experience 🎨

## **📜 Contributing**
Want to contribute? Feel free to **fork the repo** and submit a **pull request**! 🚀

## **📧 Contact**
📩 **Email:** roshanrb001@gmail.com    
💼 **LinkedIn:** https://www.linkedin.com/in/roshanbhaskar/

