from controller import Robot
import numpy as np
import random
import pickle
import os
import math


ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
Q_TABLE_FILE = "q_table.pkl"

# Config
ENABLE_BACKWARD_MC = True
BACKWARD_ALPHA_SCALE = 0.7
MC_USE_BASELINE = False
MC_LAMBDA_BLEND = 1.0

# Dims
WORLD_HALF_EXTENT = 1.5
GRID_CELLS_X = 100
GRID_CELLS_Y = 100
ORIENT_BINS = 20

# Meta
GOAL_POS = [0, -0.5, 0]
GOAL_RADIUS = 0.15

# Sensores
SENSOR_OBS_THRESHOLD = 80.0
SENSOR_HIT_THRESHOLD = 100.0
ACTIONS = 4
TRAINING_MODE = True

# =========== INIT ==============
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

compass = robot.getDevice('compass')
compass.enable(timestep)

q_table = {}

episode = 0
steps = 0
total_reward = 0.0
training_mode = TRAINING_MODE
trajectory = []



# =========== FUNC ==============
def load_q_table():
    global q_table
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, 'rb') as f:
            q_table = pickle.load(f)
        print(f"Q-table cargada: {len(q_table)} estados")
    else:
        print("Iniciando nueva Q-table")
        q_table = {}

def save_q_table():
    with open(Q_TABLE_FILE, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Q-table guardada: {len(q_table)} estados")

def angle_deg_from_compass():
    north = compass.getValues()
    rad = math.atan2(north[0], north[1])
    angle_deg = (rad * 180.0 / math.pi) % 360.0
    return angle_deg

def get_orientation_bin(angle_deg):
    sector = 360.0 / ORIENT_BINS
    bin_idx = int((angle_deg + sector / 2.0) // sector) % ORIENT_BINS
    return bin_idx

def get_orientation():
    angle_deg = angle_deg_from_compass()
    bin_idx = get_orientation_bin(angle_deg)
    return bin_idx, angle_deg

def get_direction_name(bin_idx):
    if ORIENT_BINS == 8:
        directions = ["N", "NE", "E", "SE",
                      "S", "SO", "O", "NO"]
        return directions[bin_idx]
    else:
        sector = 360.0 / ORIENT_BINS
        center_deg = (bin_idx * sector) % 360.0
        return f"Dir{bin_idx} ({center_deg:.0f}°)"

def discretize_pos_to_grid(x, y):
    xn = (x + WORLD_HALF_EXTENT) / (2.0 * WORLD_HALF_EXTENT)
    yn = (y + WORLD_HALF_EXTENT) / (2.0 * WORLD_HALF_EXTENT)
    zx = int(xn * GRID_CELLS_X)
    zy = int(yn * GRID_CELLS_Y)
    zx = max(0, min(GRID_CELLS_X - 1, zx))
    zy = max(0, min(GRID_CELLS_Y - 1, zy))
    return zx, zy

def get_state():
    readings = [s.getValue() for s in distance_sensors]
    front = int(readings[0] > SENSOR_OBS_THRESHOLD or readings[7] > SENSOR_OBS_THRESHOLD)
    left  = int(readings[5] > SENSOR_OBS_THRESHOLD or readings[6] > SENSOR_OBS_THRESHOLD)
    right = int(readings[1] > SENSOR_OBS_THRESHOLD or readings[2] > SENSOR_OBS_THRESHOLD)
    pos = gps.getValues()      # [x, y, z]
    zone_x, zone_y = discretize_pos_to_grid(pos[0], pos[1])
    orient_bin, _ = get_orientation()
    return (front, left, right, zone_x, zone_y, orient_bin)

def get_reward():
    pos = gps.getValues()
    distance = np.sqrt((pos[0] - GOAL_POS[0])**2 + (pos[1] - GOAL_POS[1])**2)
    if distance < GOAL_RADIUS:
        return 100
    readings = [s.getValue() for s in distance_sensors]
    if max(readings) > SENSOR_HIT_THRESHOLD:
        return -10
    if abs(pos[0]) > WORLD_HALF_EXTENT or abs(pos[1]) > WORLD_HALF_EXTENT:
        return -15
    return -distance * 2.0

def ensure_state(s):
    if s not in q_table:
        q_table[s] = np.zeros(ACTIONS, dtype=float)

def choose_action(state):
    if training_mode:
        epsilon = max(0.05, EPSILON * (0.998 ** episode))
    else:
        epsilon = 0.0
    if random.random() < epsilon:
        return random.randint(0, ACTIONS - 1)
    ensure_state(state)
    return int(np.argmax(q_table[state]))

def execute_action(action):
    MAX_SPEED = 6.28
    if action == 0:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)
    elif action == 1:
        left_motor.setVelocity(MAX_SPEED * 0.3)
        right_motor.setVelocity(MAX_SPEED)
    elif action == 2:
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED * 0.3)
    elif action == 3:
        left_motor.setVelocity(-MAX_SPEED * 0.5)
        right_motor.setVelocity(-MAX_SPEED * 0.5)

def stop_robot():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

def apply_backward_returns_mc(trajectory):
    if not trajectory:
        print("No backward")
        return

    alpha_back = ALPHA * BACKWARD_ALPHA_SCALE
    G = 0.0

    for (s, a, r, s_next) in reversed(trajectory):
        G = r + GAMMA * G
        ensure_state(s)
        q_sa = q_table[s][a]

        if MC_USE_BASELINE:
            target = G
        else:
            target = (1.0 - MC_LAMBDA_BLEND) * q_sa + MC_LAMBDA_BLEND * G

        q_table[s][a] = q_sa + alpha_back * (target - q_sa)

def reset_episode():
    global steps, total_reward, episode, trajectory
    max_states = 8 * (GRID_CELLS_X * GRID_CELLS_Y) * ORIENT_BINS
    coverage = (len(q_table) / max_states) * 100.0

    print(f"\n{'='*60}")
    print(f"EPISODIO {episode} COMPLETADO")
    print(f"   Pasos: {steps}")
    print(f"   Recompensa total: {total_reward:.2f}")
    print(f"   Estados explorados: {len(q_table)} de {max_states}")
    print(f"   Cobertura: {coverage:.3f}%")
    print(f"{'='*60}\n")

    episode += 1
    steps = 0
    total_reward = 0.0
    trajectory = []

    if episode % 10 == 0:
        save_q_table()


# =========== LOOP ==============
load_q_table()
while robot.step(timestep) != -1:
    state = get_state()
    action = choose_action(state)
    execute_action(action)

    for _ in range(5):
        robot.step(timestep)

    next_state = get_state()
    reward = get_reward()
    total_reward += reward
    steps += 1

    trajectory.append((state, action, reward, next_state))

    if training_mode:
        ensure_state(state)
        ensure_state(next_state)
        td_target = reward + GAMMA * np.max(q_table[next_state])
        q_table[state][action] += ALPHA * (td_target - q_table[state][action])

    if steps % 50 == 0:
        pos = gps.getValues()
        distance = np.sqrt((pos[0] - GOAL_POS[0])**2 + (pos[1] - GOAL_POS[1])**2)
        orient_bin, angle_deg = get_orientation()
        dir_name = get_direction_name(orient_bin)
        mode = "TRAIN" if training_mode else "EVAL"
        action_names = ["Front", "Left", "Right", "Back"]

        print(f"[{mode}] Paso {steps}: Pos({pos[0]:.2f},{pos[1]:.2f}) "
              f"Orient={dir_name} ({angle_deg:.1f}°) "
              f"Acción={action_names[action]} Dist={distance:.2f} R={reward:.2f}")

    if reward == 100:
        print("YASTAMOS META ALCANZADA")
        stop_robot()

        if training_mode and ENABLE_BACKWARD_MC:
            print("Backward MC: propagando retornos hacia atrás…")
            apply_backward_returns_mc(trajectory)

        if training_mode:
            save_q_table()
            print(f"Episodio {episode} completado exitosamente")
            print("Robot detenido. Reinicia la simulación para continuar entrenando.")
        else:
            print("Evaluación exitosa - Robot aprendió correctamente")
        break

    if steps >= 1000:
        print("Timeout - Episodio fallido")
        stop_robot()

        if training_mode and ENABLE_BACKWARD_MC:
            print("Backward MC (timeout): propagando retornos hacia atrás…")
            apply_backward_returns_mc(trajectory)

        if training_mode:
            save_q_table()
            print("Robot detenido. Reinicia la simulación para continuar.")
        else:
            print("No alcanzó la meta en modo evaluación")
        break

if training_mode:
    save_q_table()
    print("\nSimulación finalizada - Q-table guardada")
