import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import random
from vision_model import EngagementCNN

class EducationEnv:
    def __init__(self):
        # State: [Engagement (0-1), Performance (0-100), TimeOnTask (norm), Difficulty (0-1)]
        self.state_dim = 4
        
        # Actions: 0: Video Lecture, 1: Interactive Quiz, 2: Reading Material, 3: Hard Project
        self.action_space = ["Video Lecture", "Interactive Quiz", "Reading Material", "Hard Project"]
        self.action_dim = len(self.action_space)
        
        self.current_state = np.array([0.5, 50.0, 0.0, 0.5]) # Initial state
        
        # --- VISION MODEL INTEGRATION ---
        self.device = torch.device('cpu')
        self.vision_model = EngagementCNN(num_classes=3)
        try:
            self.vision_model.load_state_dict(torch.load('engagement_model.pth', map_location=self.device))
            self.vision_model.eval()
            print("Vision Model Loaded Successfully!")
        except Exception as e:
            print(f"Warning: Vision Model not found ({e}). Simulating vision.")
            self.vision_model = None

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Dataset for random sampling (Simulating Webcam)
        self.data_dir = 'data/Student-engagement-dataset/Engaged'
        self.image_files = []
        if os.path.exists(self.data_dir):
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_files.append(os.path.join(root, file))
        else:
            print(f"Warning: Data directory {self.data_dir} not found.")

    def reset(self):
        self.current_state = np.array([0.5, 50.0, 0.0, 0.5])
        return self.current_state

    def get_visual_engagement(self):
        """Simulates a webcam feed by sampling from the dataset and running the CNN."""
        if not self.vision_model or not self.image_files:
            return np.random.uniform(0.4, 0.9) # Fallback
        
        # 1. Sample Random Image
        img_path = random.choice(self.image_files)
        
        # 2. Process Image
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 3. Inference
            with torch.no_grad():
                output = self.vision_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                
            # Mapping: 0:Confused, 1:Engaged, 2:Frustrated (Based on alphabetical)
            # We want Engagement to be high if class is 1 (Engaged)
            # If Confused (0) or Frustrated (2), engagement is lower.
            
            # Let's assume indices from class_map.txt
            # If alphabetic: Confused=0, Engaged=1, Frustrated=2
            p_confused = probs[0][0].item()
            p_engaged = probs[0][1].item()
            p_frustrated = probs[0][2].item()
            
            engagement_score = (p_engaged * 1.0) + (p_confused * 0.4) + (p_frustrated * 0.2)
            return engagement_score
        except Exception as e:
            print(f"Error in vision inference: {e}")
            return 0.5

    def step(self, action, user_feedback):
        """
        Simulate the environment response with VISION FUSION.
        """
        prev_performance = self.current_state[1]
        
        # 1. Get Visual Engagement
        visual_score = self.get_visual_engagement()
        
        # 2. Get Manual Feedback (optional, if user overrides)
        manual_engagement = user_feedback.get('engagement', 0.5)
        
        # 3. Fuse: 70% Vision, 30% Manual
        final_engagement = (visual_score * 0.7) + (manual_engagement * 0.3)
        final_engagement = np.clip(final_engagement, 0.0, 1.0)

        performance = user_feedback.get('score', prev_performance)
        time_spent = user_feedback.get('time', 0.1) 
        
        # Heuristic: Difficulty adjusts
        difficulty = self.current_state[3]
        if action == 3: # Hard Project
            difficulty = min(1.0, difficulty + 0.1)
        elif action == 0: # Video
            difficulty = max(0.0, difficulty - 0.05)

        next_state = np.array([final_engagement, performance, time_spent, difficulty])
        self.current_state = next_state
        
        # Reward Calculation
        delta_score = performance - prev_performance
        reward = (delta_score * 0.5) + (final_engagement * 10) - (time_spent * 2)
        
        done = False 
        
        return next_state, reward, done, {'visual_score': visual_score}
