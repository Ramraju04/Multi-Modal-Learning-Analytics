from flask import Flask, render_template, jsonify, request
import numpy as np
from agent import Agent
from environment import EducationEnv
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize Environment and Agent
env = EducationEnv()
agent = Agent(state_dim=env.state_dim, action_dim=env.action_dim)

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/api/reset', methods=['POST'])
def reset():
    state = env.reset()
    return jsonify({'state': state.tolist()})

@app.route('/api/action', methods=['POST'])
def get_action():
    data = request.json
    state = np.array(data['state'])
    
    # Get action from DRL agent
    action_idx = agent.get_action(state)
    action_name = env.action_space[action_idx]
    
    return jsonify({
        'action_idx': action_idx,
        'action_name': action_name
    })

@app.route('/api/step', methods=['POST'])
def step():
    data = request.json
    action_idx = data['action_idx']
    current_state = np.array(data['state'])
    
    # User feedback from frontend (simulating the 'sensor' data)
    user_feedback = {
        'engagement': float(data['engagement']),
        'score': float(data['score']),
        'time': float(data['time'])
    }
    
    # Step environment
    next_state, reward, done, info = env.step(action_idx, user_feedback)
    
    # Train Agent ONCE (Online Learning)
    agent.remember(current_state, action_idx, reward, next_state, done)
    agent.train()
    
    return jsonify({
        'next_state': next_state.tolist(),
        'reward': reward,
        'done': done,
        'visual_score': info.get('visual_score', 0.5)
    })

if __name__ == '__main__':
    app.run(debug=True)
