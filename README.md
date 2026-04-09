# 🚀 Multi-Modal Learning Analytics System

An intelligent AI-powered learning platform that adapts educational content based on user engagement, performance, and behavioral signals in real-time.

---

## 📌 Overview

This project simulates a **personalized learning system** that dynamically adjusts content difficulty and type (e.g., video lectures, exercises) based on:

- User engagement level
- Assessment performance
- AI-based decision making

The system uses **machine learning + web interface** to create an adaptive learning experience.

---

## 🎯 Key Features

✅ Real-time learning adaptation  
✅ AI-based decision making (Agent)  
✅ Interactive dashboard visualization  
✅ Engagement simulation system  
✅ Personalized content recommendation  
✅ Flask-based web interface  

---

## 🧠 How It Works

1. User inputs:
   - Engagement level
   - Assessment score  

2. AI Agent analyzes:
   - Current learning state
   - Difficulty level
   - Reward system  

3. System outputs:
   - Recommended content (Video / Practice)
   - Updated learning metrics  

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **Machine Learning:** PyTorch  
- **Data Processing:** NumPy  
- **Image Processing:** Pillow  

---

## 📁 Project Structure


MultiED/
│
├── app.py # Flask main app
├── agent.py # AI decision logic
├── environment.py # Simulation environment
├── train_cnn.py # Model training
├── vision_model.py # Vision-based model
├── class_map.txt # Class labels
├── requirements.txt # Dependencies
│
├── templates/ # HTML files
├── static/ # CSS & JS
├── data/ # Dataset (if any)


---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

cd Multi-Modal-Learning-Analytics
2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the application
python app.py
5️⃣ Open in browser
http://127.0.0.1:5000
