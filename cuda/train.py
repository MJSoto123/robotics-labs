from controller import Supervisor
import numpy as np
import random
import pickle
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = False  
FORCE_CPU = True  

def setup_cuda():
    if FORCE_CPU:
        device = torch.device("cpu")
        print("CPU FORZADO (FORCE_CPU=True)")
        return device
    
    if torch.cuda.is_available() and USE_CUDA:
        device = torch.device("cuda")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        print("USANDO CUDA (GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

device = setup_cuda()

LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


if device.type == 'cuda':
    BATCH_SIZE = 64  
    MEMORY_SIZE = 10000
    print("Batch size: 64 (GPU)")
else:
    BATCH_SIZE = 32  
    MEMORY_SIZE = 5000  
    print("Batch size: 32 (CPU)")

UPDATE_TARGET_EVERY = 100
GOAL_POS = [0, -0.5, 0]
ACTIONS = 4

# Robot Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
node = robot.getSelf()  
translation_field = node.getField("translation")
rotation_field = node.getField("rotation")

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Sensores
distance_sensors = []
for i in range(8):
    sensor = robot.getDevice(f'ps{i}')
    sensor.enable(timestep)
    distance_sensors.append(sensor)

# GPS
gps = robot.getDevice('gps')
gps.enable(timestep)

# Giroscopio
try:
    inertial_unit = robot.getDevice('inertial unit')
    inertial_unit.enable(timestep)
    has_imu = True
except:
    has_imu = False
    print("IMU no disponible")


episode = 0
steps = 0
total_reward = 0
training_mode = True
global_step = 0

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_size)
        )
        
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Tensores en GPU"""
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
        
        
        self.training_steps = 0
        
        print(f"Agente inicializado en: {device}")
        print(f"Parámetros del modelo: {sum(p.numel() for p in self.model.parameters())}")
    
    def get_state_vector(self):
        """Estado actual"""
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
            sensor_values +           
            pos_normalized +          
            [dx_n, dy_n] +           
            [distance, angle] +      
            [orientation]            
        )
        
        return state
    
    def act(self, state):
        """Epsilon-greedy"""
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
            
            self.training_steps += 1
            
            return loss.item()
            
        except Exception as e:
            print(f"Error: {e}")
            return 0.0
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'training_steps': self.training_steps,
            'device': str(device)
        }
        torch.save(checkpoint, filename)
        print(f"Modelo guardado: {filename}")
    
    def load(self, filename):
        global episode
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename, map_location=device)
                
                expected_input_size = checkpoint['model_state_dict']['network.0.weight'].shape[1]
                if expected_input_size != self.state_size:
                    print(f"Error: espera {expected_input_size}D pero usa {self.state_size}D")
                    return False
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint.get('target_model_state_dict', 
                                                                 checkpoint['model_state_dict']))
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', EPSILON_MIN)
                episode = checkpoint.get('episode', 0)
                self.training_steps = checkpoint.get('training_steps', 0)
                
                self.memory = ReplayBuffer(MEMORY_SIZE)
                
                saved_device = checkpoint.get('device', 'cpu')
                print(f"Modelo cargado: {filename}")
                print(f" Episodio: {episode}")
                print(f" Epsilon: {self.epsilon:.3f}")
                print(f" Guardado en: {saved_device} → Cargado en: {device}")
                
                return True
                
            except Exception as e:
                print(f"Error cargando modelo: {e}")
                return False
        return False
    
    def get_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return f"GPU Mem: {allocated:.2f}GB / {reserved:.2f}GB"
        return "CPU mode"


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
        angle_diff = np.abs(np.arctan2(np.sin(angle_to_goal - robot_angle), 
                                      np.cos(angle_to_goal - robot_angle)))
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

# Detener Robot
def stop_robot():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
   
# Reiniciar el escenario con el supervisor
def reset_robot_position():
    translation_field.setSFVec3f([0, 1, 0])
    rotation_field.setSFRotation([0, 0, 1, -1.58])
    robot.step(timestep)

STATE_SIZE = 15  
agent = DQNAgent(STATE_SIZE, ACTIONS)

MODEL_FILE = "dqn_model_cuda.pth"
if not agent.load(MODEL_FILE):
    print("Empezando desde cero")


print("\nInicializando sensores...")
for _ in range(10):
    robot.step(timestep)
print("Sensores listos\n")

losses = []
episode_rewards = []

print("Iniciando entrenamiento con Deep Q-Network")
print("=" * 60)

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
            print(f"Network actualizada (paso {global_step}) - Loss: {avg_loss:.4f} - {agent.get_memory_usage()}")
    
    
    if steps % 50 == 0:
        pos = gps.getValues()
        distance = np.sqrt((pos[0] - GOAL_POS[0])**2 + (pos[1] - GOAL_POS[1])**2)
        mode = "TRAIN" if training_mode else "EVAL"
        avg_loss = np.mean(losses[-50:]) if losses else 0
        action_names = ["Forward", "Left", "Right", "Turn"]
        
        print(f"[{mode}] Ep{episode:3d} Step{steps:4d}: "
              f"ε={agent.epsilon:.3f} Mem={agent.memory.size():5d} "
              f"Pos=({pos[0]:+.2f},{pos[1]:+.2f}) Dist={distance:.2f} "
              f"R={reward:+.1f} Loss={avg_loss:.4f} A={action_names[action]}")
    
    
    if done or steps >= 4000:
        if done:
            print(f"META Episodio {episode} en {steps} pasos - Recompensa: {total_reward:.2f}")
            reset_robot_position()
        else:
            print(f"Timeout episodio {episode} - Recompensa: {total_reward:.2f}")
            reset_robot_position()
        
        episode_rewards.append(total_reward)
        
        if training_mode:
            agent.save(MODEL_FILE)
            agent.decay_epsilon()
            
            
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                max_reward = max(episode_rewards[-10:])
                print(f"Últimos 10 episodios: Promedio={avg_reward:.2f}, Máximo={max_reward:.2f}")
        
        
        if episode % 20 == 0 and episode > 0:
            checkpoint_file = f"dqn_checkpoint_ep{episode}.pth"
            agent.save(checkpoint_file)
            print(f"Checkpoint guardado: {checkpoint_file}")
        
        
        episode += 1
        steps = 0
        total_reward = 0
        
        if training_mode:
            for _ in range(20):
                robot.step(timestep)
        else:
            stop_robot()
            break


if training_mode:
    agent.save(MODEL_FILE)
    print("\n" + "=" * 60)
    print("Entrenamiento completado")
    print(f"   Episodios: {episode}")
    print(f"   Pasos totales: {global_step}")
    if episode_rewards:
        print(f"   Mejor recompensa: {max(episode_rewards):.2f}")
        print(f"   Promedio general: {np.mean(episode_rewards):.2f}")
    print("=" * 60)