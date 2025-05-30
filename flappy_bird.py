import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import layers
import random

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class FlappyBirdEnv:
    def __init__(self):
        self.screen_width = 400
        self.screen_height = 600
        self.pipe_gap = 150
        self.gravity = 1
        self.flap_strength = -10
        self.reset()

    def reset(self):
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        self.pipe_x = self.screen_width
        self.pipe_height = np.random.randint(100, 400)
        self.score = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        if action == 1:
            self.bird_velocity = self.flap_strength

        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        self.pipe_x -= 5
        if self.pipe_x < -50:
            self.pipe_x = self.screen_width
            self.pipe_height = np.random.randint(100, 400)
            self.score += 1

        if self.bird_y > self.screen_height or self.bird_y < 0 or (
            self.pipe_x < 50 and 
            (self.bird_y < self.pipe_height or self.bird_y > self.pipe_height + self.pipe_gap) ):
            self.done = True

        reward = 0.1  
        if self.done:
            reward = -1 

        return self.get_state(), reward, self.done

    def get_state(self):
        return np.array([self.bird_y, self.bird_velocity, self.pipe_x, self.pipe_height])

class FlappyBirdAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.best_score = -float('inf')
        self.no_improvement_count = 0

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=self.state_shape),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 5000: 
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, episode):
        wsl_path = './flappy bird/model'
        if not os.path.exists(wsl_path):
            os.makedirs(wsl_path)
        save_path = os.path.join(wsl_path, f"flappy_bird_model_episode_{episode}.h5")
        self.model.save(save_path)
        logging.info(f"Model saved at {save_path}")

if __name__ == "__main__":
    env = FlappyBirdEnv()
    state_shape = (4,)
    action_space = 2
    agent = FlappyBirdAgent(state_shape, action_space)
    episodes = 1000
    early_stopping_threshold = 20

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 4])
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                logging.info(f"Episode: {e+1}/{episodes}, Score: {total_reward}")
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}")

                if total_reward > agent.best_score:
                    agent.best_score = total_reward
                    agent.no_improvement_count = 0
                else:
                    agent.no_improvement_count += 1

                if agent.no_improvement_count >= early_stopping_threshold:
                    logging.info(f"No improvement for {early_stopping_threshold} episodes. Early stopping triggered.")
                    agent.save_model(e + 1)
                    print("Early stopping triggered. Training stopped.")
                    exit()
                break

        agent.replay()

        if (e + 1) % 50 == 0: 
            agent.save_model(e + 1)
