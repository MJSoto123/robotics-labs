from controller import Robot
import numpy as np
import random
import pickle
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
UPDATE_TARGET_EVERY = 100
GOAL_POS = [0, -0.5, 0]
ACTIONS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

robot = Robot()
timestep = int(robot.getBasicTimeStep())
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

distance_sensors = []
for i in range(8):
    sensor = robot.getDevice(f'ps{i}')
    sensor.enable(timestep)
    distance_sensors.append(sensor)

gps = robot.getDevice('gps')
gps.enable(timestep)

try:
    inertial_unit = robot.getDevice('inertial unit')
    inertial_unit.enable(timestep)
    has_imu = True
except:
    has_imu = False

episode = 0
steps = 0
total_reward = 0
training_mode = True
global_step = 0

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )
    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON
        self.model = DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(MEMORY_SIZE)
    def get_state_vector(self):
        sensor_values = [s.getValue() / 4096.0 for s in distance_sensors]
        pos = gps.getValues()
        pos_normalized = [(pos[0] + 1.5) / 3.0, (pos[1] + 1.5) / 3.0]
    
        dx = pos[0] - GOAL_POS[0]
        dy = pos[1] - GOAL_POS[1]
        distance = np.sqrt(dx**2 + dy**2) / 3.0
        angle = np.arctan2(dy, dx) / np.pi
    
        dx_n = np.clip(dx / 1.5, -1.0, 1.0)
        dy_n = np.clip(dy / 1.5, -1.0, 1.0)
    
        if has_imu:
            rpy = inertial_unit.getRollPitchYaw()
            orientation = rpy[2] / np.pi
        else:
            orientation = 0.0
    
        state = np.array(
            sensor_values
            + pos_normalized
            + [dx_n, dy_n
            + [distance, angle]
            + [orientation]
        )
        return state

    def act(self, state):
        if training_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    def replay(self):
        if self.memory.size() < BATCH_SIZE * 2:
            return 0.0
        try:
            states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
            if states.shape[1] != self.state_size:
                return 0.0
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * GAMMA * next_q
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return loss.item()
        except Exception as e:
            print(f"Error en replay: {e}")
            return 0.0
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode
        }
        torch.save(checkpoint, filename)
        print(f"Modelo guardado: {filename}")
    def load(self, filename):
        global episode
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            expected_input_size = checkpoint['model_state_dict']['fc1.weight'].shape[1]
            if expected_input_size != self.state_size:
                print(f"Modelo incompatible: espera {expected_input_size}D pero usa {self.state_size}D")
                return False
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', EPSILON_MIN)
            episode = checkpoint.get('episode', 0)
            self.target_model.load_state_dict(self.model.state_dict())
            self.memory = ReplayBuffer(MEMORY_SIZE)
            print(f"Modelo cargado: {filename} (Episodio {episode})")
            return True
        return False

def get_reward():
    pos = gps.getValues()
    distance = np.sqrt((pos[0] - GOAL_POS[0])**2 + (pos[1] - GOAL_POS[1])**2)
    if distance < 0.15:
        return 100.0, True
    readings = [s.getValue() for s in distance_sensors]
    if max(readings) > 100:
        return -5.0, False
    if abs(pos[0]) > 1.45 or abs(pos[1]) > 1.45:
        return -10.0, False
    distance_reward = -distance * 1.5
    if has_imu:
        dx = GOAL_POS[0] - pos[0]
        dy = GOAL_POS[1] - pos[1]
        angle_to_goal = np.arctan2(dy, dx)
        rpy = inertial_unit.getRollPitchYaw()
        robot_angle = rpy[2]
        angle_diff = np.abs(np.arctan2(np.sin(angle_to_goal - robot_angle), np.cos(angle_to_goal - robot_angle)))
        orientation_bonus = 2.0 * (1.0 - angle_diff / np.pi)
    else:
        orientation_bonus = 0
    total_reward = distance_reward + orientation_bonus
    return total_reward, False

def execute_action(action):
    MAX_SPEED = 6.28
    if action == 0:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)
    elif action == 1:
        left_motor.setVelocity(MAX_SPEED * 0.2)
        right_motor.setVelocity(MAX_SPEED)
    elif action == 2:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED * 0.2)
    elif action == 3:
        left_motor.setVelocity(-MAX_SPEED * 0.5)
        right_motor.setVelocity(MAX_SPEED * 0.5)

def stop_robot():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

STATE_SIZE = 15
agent = DQNAgent(STATE_SIZE, ACTIONS)
MODEL_FILE = "dqn_model_v2.pth"
if not agent.load(MODEL_FILE):
    print("No se encontró modelo previo, empezando desde cero")

print("Iniciando Deep Q-Network (PyTorch)")
for _ in range(10):
    robot.step(timestep)
print("Sensores listos")

losses = []
episode_rewards = []

while robot.step(timestep) != -1:
    state = agent.get_state_vector()
    action = agent.act(state)
    execute_action(action)
    for _ in range(5):
        robot.step(timestep)
    next_state = agent.get_state_vector()
    reward, done = get_reward()
    total_reward += reward
    steps += 1
    global_step += 1
    if training_mode:
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()
        if loss > 0:
            losses.append(loss)
        if global_step % UPDATE_TARGET_EVERY == 0:
            agent.update_target_network()
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Target network actualizada paso {global_step} - Loss promedio: {avg_loss:.4f}")
    if steps % 50 == 0:
        pos = gps.getValues()
        distance = np.sqrt((pos[0] - GOAL_POS[0])**2 + (pos[1] - GOAL_POS[1])**2)
        mode = "TRAIN" if training_mode else "EVAL"
        avg_loss = np.mean(losses[-50:]) if losses else 0
        action_names = ["Forward", "Left", "Right", "Turn"]
        print(f"[{mode}] Ep{episode:3d} Step{steps:4d}: ε={agent.epsilon:.3f} Mem={agent.memory.size():5d} Pos=({pos[0]:+.2f},{pos[1]:+.2f}) Dist={distance:.2f} R={reward:+.1f} Loss={avg_loss:.4f} A={action_names[action]}")
    if done or steps >= 1000:
        if done:
            print(f"Meta alcanzada en {steps} pasos Recompensa total: {total_reward:.2f}")
        else:
            print(f"Timeout episodio {episode}")
        episode_rewards.append(total_reward)
        if training_mode:
            agent.save(MODEL_FILE)
            agent.decay_epsilon()
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Promedio últimos 10 episodios: {avg_reward:.2f}")
        episode += 1
        steps = 0
        total_reward = 0
        if training_mode:
            for _ in range(20):
                robot.step(timestep)
        else:
            stop_robot()
            break
        if episode % 20 == 0:
            agent.save(f"dqn_checkpoint_ep{episode}.pth")

if training_mode:
    agent.save(MODEL_FILE)
    print(f"Entrenamiento completado Episodios: {episode} Pasos totales: {global_step}")
    if episode_rewards:
        print(f"Mejor recompensa: {max(episode_rewards):.2f}")