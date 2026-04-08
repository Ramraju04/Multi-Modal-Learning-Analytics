# Multi-Modal Learning Analytics (MMLA) using Deep Reinforcement Learning

Welcome to the **MMLA Project**. This system is an advanced educational technology platform that uses Artificial Intelligence (AI) to personalize learning content for students in real-time.

---

## 🚀 How to Run the Project

### Prerequisites
- **Python 3.8+** installed on your system.
- Basic understanding of using the command line/terminal.

### Step-by-Step Execution

1.  **Open Terminal/Command Prompt**
    Navigate to the project folder:
    ```bash
    cd "path\to\your\project\MultiED"
    ```

2.  **Install Dependencies** (if not already installed)
    ```bash
    pip install flask numpy torch
    ```

3.  **Run the Server**
    Execute the Python script to start the Flask backend:
    ```bash
    python app.py
    ```

4.  **Access the Web Application**
    Open your web browser (Chrome/Edge/Firefox) and go to:
    👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 🧠 How This Project Works

The system operates as a **Closed-Loop Adaptive System**:

1.  **Sensors (Simulated):** The system collects data points representing the student's state:
    - **Engagement Level:** Is the student paying attention? (0.0 to 1.0)
    - **Performance Score:** How well did they do on the last task? (0-100)
    - **Time on Task:** How fast are they learning?

2.  **Fusion Layer:** These diverse inputs are combined into a single "State Vector".

3.  **The AI Brain (DQN Agent):**
    - A **Deep Q-Network (Deep Reinforcement Learning)** analyzes the State Vector.
    - It predicts which learning action (Video, Quiz, Reading, Project) will yield the **highest long-term reward**.
    - The "Reward" is calculated based on maintaining high engagement and improving test scores.

4.  **Action & Feedback:** The selected content is presented to the student. The new engagement/score data is fed back into the system, and the AI updates its internal model ("learns") to be better next time.

---

## 🌟 Why is this Useful?

- **Personalization at Scale:** Traditional classrooms treat everyone the same. MMLA treats every student as a unique individual.
- **Real-Time Adaptation:** It fixes boredom or confusion *instantly*, not after the exam is failed.
- **Data-Driven Insights:** Provides educators with deep analytics on *how* students learn, not just *what* they know.

---

## 📊 How the Output is Shown

The primary output is visualized on the **Dashboard Page**:

1.  **Live Charts:** A line graph plots the "Estimated Reward" (Learning Gain) and "Content Difficulty" over time.
    - *Green Line:* Shows how much the student is learning.
    - *Purple Line:* Shows the difficulty level adapting (up or down).

2.  **AI Recommendation:** A prominent display shows exactly what the AI suggests next (e.g., "Interactive Quiz").

3.  **Event Logs:** A scrolling terminal view shows the backend decision process step-by-step.

---

## 🛠 Project Structure

- **`app.py`**: The main Flask server and API endpoints.
- **`agent.py`**: The Deep Reinforcement Learning (DQN) model logic.
- **`environment.py`**: Simulates the student's response to different teaching methods.
- **`templates/`**: HTML files for the website (Home, About, Dashboard, etc.).
- **`static/css/style.css`**: The "Aurora" design system styling.
- **`static/script.js`**: Frontend logic for the interactive dashboard.

---