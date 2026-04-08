let currentState = [0.5, 50.0, 0.0, 0.5]; // Initial State
let stepCount = 0;

// Initialize Chart
const ctx = document.getElementById('learningCurveChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Reward (Learning Gain)',
            data: [],
            borderColor: '#64ffda',
            backgroundColor: 'rgba(100, 255, 218, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 4
        }, {
            label: 'Content Difficulty',
            data: [],
            borderColor: '#bd34fe',
            borderDash: [5, 5],
            tension: 0.4,
            pointRadius: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#ccd6f6' }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                ticks: { color: '#8892b0' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#8892b0' }
            }
        },
        animation: {
            duration: 800,
            easing: 'easeOutQuart'
        }
    }
});

function log(msg) {
    const list = document.getElementById('log-list');
    const item = document.createElement('li');
    item.className = 'log-item';

    const time = new Date().toLocaleTimeString();
    item.innerHTML = `<span style="color: #64ffda">[${time}]</span> Step ${stepCount}: ${msg}`;

    list.prepend(item);
}

// UI Controls
const engSlider = document.getElementById('eng-slider');
const engValue = document.getElementById('eng-value');

if (engSlider) {
    engSlider.addEventListener('input', (e) => {
        engValue.textContent = e.target.value;
    });
}

// Main Interaction
async function takeStep() {
    const button = document.getElementById('next-step-btn');
    button.disabled = true;
    button.innerText = "Processing...";

    // 1. Get Inputs from User (Simulating Sensors)
    const engagement = parseFloat(document.getElementById('eng-slider').value);
    const score = parseFloat(document.getElementById('score-input').value);

    // Get Action from Agent (based on PREVIOUS state)
    try {
        const actionRes = await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: currentState })
        });
        const actionData = await actionRes.json();

        // Update UI with Agent's Choice
        document.querySelector('#learning-module h3').textContent = actionData.action_name;
        document.querySelector('#learning-module p').textContent = "Agent selected this module based on your profile state.";

        // Execute Action (Step Environment)
        const stepRes = await fetch('/api/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action_idx: actionData.action_idx,
                state: currentState,
                engagement: engagement,
                score: score,
                time: 0.2
            })
        });

        const stepData = await stepRes.json();

        // Update Local State
        currentState = stepData.next_state;
        stepCount++;

        // Update Stats
        document.getElementById('reward-display').textContent = stepData.reward.toFixed(2);
        document.getElementById('difficulty-display').textContent = currentState[3].toFixed(2);
        document.getElementById('step-display').textContent = stepCount;

        // Update Engagement Visual
        const engDisplay = document.getElementById('eng-value');
        if (stepData.visual_score !== undefined) {
            engDisplay.innerHTML = `${engagement} <span style="font-size:0.7em; color: #6ee7b7;">(Vision: ${stepData.visual_score.toFixed(2)})</span>`;
        }

        log(`Client Eng: ${engagement}, Score: ${score} -> Action: <strong>${actionData.action_name}</strong> -> Reward: ${stepData.reward.toFixed(2)} | Vision: ${stepData.visual_score ? stepData.visual_score.toFixed(2) : 'N/A'}`);

        // Update Chart
        chart.data.labels.push(stepCount);
        chart.data.datasets[0].data.push(stepData.reward);
        chart.data.datasets[1].data.push(currentState[3] * 10); // Scale difficulty for visibility (0-1 -> 0-10)

        // Keep chart clean (last 20 points)
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }

        chart.update();

    } catch (error) {
        console.error("Error:", error);
        log("Error communicating with server.");
    }

    button.disabled = false;
    button.innerText = "Step Simulation";
}

async function resetEnv() {
    try {
        const res = await fetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        currentState = data.state;
        stepCount = 0;

        // Reset Chart
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.data.datasets[1].data = [];
        chart.update();

        // Reset UI
        document.getElementById('step-display').textContent = '0';
        document.getElementById('reward-display').textContent = '0.0';
        document.querySelector('#learning-module h3').textContent = "Waiting for input...";
        document.getElementById('log-list').innerHTML = '';

        log("Environment Reset.");
    } catch (e) {
        console.error(e);
    }
}

// Initial Load
document.addEventListener('DOMContentLoaded', () => {
    resetEnv();
});
