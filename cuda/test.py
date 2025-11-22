from controller import Robot
import numpy as np
import torch
import torch.nn as nn
import glob
import os

MODEL_FILE = "dqn_model_cuda.pth"  
USE_CUDA = True  

GOAL_POS = [0, -0.5, 0]
SHOW_Q_VALUES = True  

available_models = glob.glob("*.pth")
if available_models:
    print("Modelos:")
    for model in available_models:
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"   • {model} ({size_mb:.2f} MB)")
    
    if MODEL_FILE not in available_models:
        print(f"\n'{MODEL_FILE}' no encontrado")

device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
print(f"Modo demostración - Usando: {device}")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

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

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        return self.network(x)


model = DQN().to(device)
model.eval()

try:
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo cargado: {MODEL_FILE}")
    print(f"Episodios: {checkpoint.get('episode', 'N/A')}")
    print("=" * 60)
except Exception as e:
    print(f"Error: {e}")
    exit()




def get_state():
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
    
    return np.array(sensor_values + pos_normalized + [dx_n, dy_n] + 
                   [distance, angle] + [orientation])

def get_action(state):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return q_values.argmax().item(), q_values.cpu().numpy()[0]

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

def get_distance():
    pos = gps.getValues()
    return np.sqrt((pos[0] - GOAL_POS[0])**2 + (pos[1] - GOAL_POS[1])**2)

action_names = ["Adelante", "Izquierda", "Derecha", "Girar"]
steps = 0

for _ in range(10):
    robot.step(timestep)

while robot.step(timestep) != -1:
    state = get_state()
    action, q_values = get_action(state)
    execute_action(action)
    
    for _ in range(5):
        robot.step(timestep)
    steps += 1

    if steps % 20 == 0:
        pos = gps.getValues()
        dist = get_distance()
        
        print(f"Paso {steps:4d}: {action_names[action]:12s} | "
              f"Pos=({pos[0]:+.2f}, {pos[1]:+.2f}) | "
              f"Dist={dist:.3f}m", end="")
        
        if SHOW_Q_VALUES:
            q_str = " | Q=[" + ", ".join([f"{q:+.2f}" for q in q_values]) + "]"
            print(q_str)
        else:
            print()
    
    if get_distance() < 0.15:
        print(f"\nMETA ALCANZADA en {steps} pasos")
        break
    
    readings = [s.getValue() for s in distance_sensors]
    
    if steps >= 4000:
        print(f"\nTimeout - Distancia final: {get_distance():.3f}m")
        break

print("Demostración completada\n")