import pygame
import cv2
import math
import random
import numpy as np
from collections import deque

# --- ГЛОБАЛЬНЫЕ ФЛАГИ ДЛЯ СИМУЛЯЦИИ ПОЛОМКИ КАМЕР ---
left_cam_active = True
right_cam_active = True

def toggle_left_cam(event, x, y, flags, param):
    global left_cam_active
    if event == cv2.EVENT_LBUTTONDOWN:
        left_cam_active = not left_cam_active

def toggle_right_cam(event, x, y, flags, param):
    global right_cam_active
    if event == cv2.EVENT_LBUTTONDOWN:
        right_cam_active = not right_cam_active

class Neuron:
    def __init__(self, name):
        self.name = name
        self.charge = 0.0
        self.threshold = 3.0
        self.leak = 0.5
        self.refractory_ticks = 0 
        self.refractory_duration = 3 

    def update(self):
        if self.refractory_ticks > 0:
            self.refractory_ticks -= 1
            return

        self.charge -= self.leak
        if self.charge < 0:
            self.charge = 0.0
        self.charge += random.uniform(-0.2, 0.2)
    
    def add_charge(self, amount):
        if self.refractory_ticks == 0:
            self.charge += amount
            if self.charge > self.threshold * 1.5:
                self.charge = self.threshold * 1.5

class Fly:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0.0
        
        self.base_speed = 4
        self.speed = self.base_speed
        
        self.hardware_blocked = False
        self.hardware_turn_dir = "" 
        
        self.is_panicking = False
        self.panic_dir = ""

    def move(self):
        if not self.hardware_blocked:
            self.x += self.speed * math.cos(math.radians(self.angle))
            self.y += self.speed * math.sin(math.radians(self.angle))
            
            if self.x > 500: self.x = 0
            elif self.x < 0: self.x = 500
            if self.y > 500: self.y = 0
            elif self.y < 0: self.y = 500

    def update_hardware_sensors(self, left_sensor, right_sensor):
        if left_sensor and right_sensor:
            self.hardware_blocked = True
            if self.hardware_turn_dir not in ["left", "right"]:
                self.hardware_turn_dir = random.choice(["left", "right"])
            
            if self.hardware_turn_dir == "left":
                self.turn_left(10)
            else:
                self.turn_right(10)
                
        elif left_sensor:
            self.hardware_blocked = True
            self.hardware_turn_dir = ""
            self.turn_right(10)
            
        elif right_sensor:
            self.hardware_blocked = True
            self.hardware_turn_dir = ""
            self.turn_left(10)
            
        else:
            self.hardware_blocked = False
            self.hardware_turn_dir = ""

    def update_panic_state(self, left_mov, right_mov):
        PANIC_TRIGGER = 15.0  
        FREE_ROAD = 2.0       
        
        if not self.is_panicking:
            if left_mov > PANIC_TRIGGER or right_mov > PANIC_TRIGGER:
                self.is_panicking = True
                self.speed = self.base_speed * 2
                
                if left_mov > right_mov:
                    self.panic_dir = "right"
                else:
                    self.panic_dir = "left"
        else:
            if not self.hardware_blocked:
                if self.panic_dir == "left":
                    self.turn_left(30)
                else:
                    self.turn_right(30)
                
            if left_mov < FREE_ROAD and right_mov < FREE_ROAD:
                self.is_panicking = False
                self.speed = self.base_speed  

    def draw(self, screen):
        color = (255, 50, 50) if self.is_panicking else (0, 255, 0)
        
        # Строгая типизация int() для совместимости со всеми версиями Pygame
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 20)
        nose_x = self.x + 30 * math.cos(math.radians(self.angle))
        nose_y = self.y + 30 * math.sin(math.radians(self.angle))
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(nose_x), int(nose_y)), 3)

    def turn_left(self, deg=10):
        self.angle -= deg  

    def turn_right(self, deg=10):
        self.angle += deg  


# --- 1. ИНИЦИАЛИЗАЦИЯ PYGAME ---
pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("НРК: Расширенная Модель")
clock = pygame.time.Clock()
fly = Fly(250, 250)

left_brain = Neuron("Left Brain")
right_brain = Neuron("Right Brain")

# --- 2. БУФЕР ДЛЯ ГРАФИКА ---
history_length = 500
left_history = deque([0.0] * history_length, maxlen=history_length)
right_history = deque([0.0] * history_length, maxlen=history_length)

# --- 3. ИНИЦИАЛИЗАЦИЯ OPENCV (С БРОНЕБОЙНОЙ ЗАЩИТОЙ) ---
print("Подключение к камере...")
cap = cv2.VideoCapture(0)

# Ждем, пока камера реально не прогреется и не выдаст первый кадр
ret, prev_frame = cap.read()
while not ret:
    print("Ожидание сигнала от камеры...")
    ret, prev_frame = cap.read()
    cv2.waitKey(100)

print("Камера успешно подключена!")
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Left Eye")
cv2.setMouseCallback("Left Eye", toggle_left_cam)

cv2.namedWindow("Right Eye")
cv2.setMouseCallback("Right Eye", toggle_right_cam)

cv2.namedWindow("Brain Telemetry Graph")

running = True
while running:
    # --- A. OPENCV ВВОД ---
    ret, frame = cap.read()
    if not ret:
        # Если кадр потерялся, не убиваем программу, а просто ждем следующий
        continue
    
    height, width, _ = frame.shape
    mid = width // 2
    
    new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    diff = cv2.absdiff(prev_frame_gray, new_frame_gray)
    
    left_diff = diff[:, 0:mid]
    right_diff = diff[:, mid:width]   
    
    # --- B. СИМУЛЯЦИЯ СБОЯ ---
    if left_cam_active:
        left_movement = cv2.mean(left_diff)[0]
    else:
        left_movement = 0.0
        left_diff = np.zeros((height, mid), dtype=np.uint8) # Безопасная очистка
        cv2.putText(left_diff, "LEFT OFF", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    if right_cam_active:
        right_movement = cv2.mean(right_diff)[0]
    else:
        right_movement = 0.0
        right_diff = np.zeros((height, width - mid), dtype=np.uint8) # Безопасная очистка
        cv2.putText(right_diff, "RIGHT OFF", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    # --- C. АППАРАТНЫЕ ПРЕПЯТСТВИЯ (КЛАВИАТУРА) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False 

    # ВАЖНО: Клавиатура считывается, только если окно Pygame активно!
    keys = pygame.key.get_pressed()
    fly.update_hardware_sensors(keys[pygame.K_LEFT], keys[pygame.K_RIGHT])
    
    # --- D. НЕЙРОМОРФНОЕ ЯДРО ---
    fly.update_panic_state(left_movement, right_movement)
    
    if not fly.is_panicking and not fly.hardware_blocked:
        if left_movement > 1.0:
            left_brain.add_charge(left_movement * 0.1)
        if right_movement > 1.0:
            right_brain.add_charge(right_movement * 0.1)

        if left_brain.charge >= left_brain.threshold:
            fly.turn_left(10)
            left_brain.charge = 0.0
            left_brain.refractory_ticks = left_brain.refractory_duration
            right_brain.charge -= 1.0  
            
        if right_brain.charge >= right_brain.threshold:
            fly.turn_right(10)
            right_brain.charge = 0.0
            right_brain.refractory_ticks = right_brain.refractory_duration
            left_brain.charge -= 1.0   
            
    left_brain.update()
    right_brain.update()
    
    # --- E. ТЕЛЕМЕТРИЯ ---
    left_history.append(left_brain.charge)
    right_history.append(right_brain.charge)
    
    graph_img = np.zeros((200, 500, 3), dtype=np.uint8)
    
    thresh_y = 200 - int(left_brain.threshold * 40)
    cv2.line(graph_img, (0, thresh_y), (500, thresh_y), (0, 255, 0), 1)
    cv2.putText(graph_img, "THRESHOLD", (10, thresh_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    for i in range(1, history_length):
        y1_l = 200 - int(left_history[i-1] * 40)
        y2_l = 200 - int(left_history[i] * 40)
        cv2.line(graph_img, (i-1, y1_l), (i, y2_l), (0, 0, 255), 1) 
        
        y1_r = 200 - int(right_history[i-1] * 40)
        y2_r = 200 - int(right_history[i] * 40)
        cv2.line(graph_img, (i-1, y1_r), (i, y2_r), (255, 0, 0), 1) 
    
    # --- F. ОБНОВЛЕНИЕ СЦЕНЫ ---
    fly.move()
    screen.fill((30, 30, 30))
    fly.draw(screen)
    pygame.display.flip()
    clock.tick(30)
    
    prev_frame_gray = new_frame_gray
    cv2.imshow("Left Eye", left_diff)
    cv2.imshow("Right Eye", right_diff)
    cv2.imshow("Brain Telemetry Graph", graph_img)
    
    # Обязательный костыль для работы окон и мыши в OpenCV
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()