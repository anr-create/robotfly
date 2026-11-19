import cv2
import numpy as np
import simka
import random
import sys
import math

# --- 1. КЛАСС НЕЙРОНА ---
class LIFNeuron:
    def __init__(self, threshold: float = 0.8, leak: float = 0.08, refractory_ticks: int = 2):
        self.threshold = threshold
        self.leak = leak
        self.refractory_ticks = refractory_ticks
        self.charge = 0.0
        self.refractory_counter = 0

    @property
    def is_ready(self) -> bool:
        return self.refractory_counter == 0

    def update(self, input_signal: float) -> int:
        if not self.is_ready:
            self.refractory_counter -= 1
            return 0

        self.charge = max(0.0, self.charge - self.leak)
        self.charge += input_signal

        if self.charge >= self.threshold:
            self.charge = 0.0
            self.refractory_counter = self.refractory_ticks
            return 1  # ⚡ Спайк!
        return 0

# --- 2. ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ---
simka.init()
WIDTH, HEIGHT = 900, 700
screen = simka.display.set_mode((WIDTH, HEIGHT))
simka.display.set_caption("RoboFly: Плавная Био-симуляция")
clock = simka.time.Clock()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Ошибка: Веб-камера не найдена!")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

left_neuron = LIFNeuron(threshold=0.8, leak=0.08, refractory_ticks=2)
right_neuron = LIFNeuron(threshold=0.8, leak=0.08, refractory_ticks=2)

fly_x, fly_y = WIDTH // 2, HEIGHT // 2
fly_angle = 0.0
font = simka.font.SysFont("Arial", 16)
prev_gray = None

# Настройки плавности
SMOOTHING = 0.15
BASE_SPEED = 1.2
BOOST_SPEED = 3.5
smooth_v_left = 0.0
smooth_v_right = 0.0

# --- 3. ГЛАВНЫЙ ЦИКЛ (Game Loop 30 FPS) ---
running = True
while running:
    for event in simka.event.get():
        if event.type == simka.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    # Обработка зрения (Сжатие и обесцвечивание)
    small_frame = cv2.resize(frame, (160, 120))
    # Небольшое размытие убирает цифровой шум матрицы камеры
    gray = cv2.GaussianBlur(cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    if prev_gray is None:
        prev_gray = gray
        continue

    # Вычитание кадров для детекции резких движений (Мухобойка)
    diff = cv2.absdiff(prev_gray, gray)
    prev_gray = gray

    # Разделение зрения на левый и правый сектора
    light_left = np.mean(gray[:, :80]) / 255.0
    light_right = np.mean(gray[:, 80:]) / 255.0
    motion_left = np.mean(diff[:, :80]) / 255.0 * 5.0
    motion_right = np.mean(diff[:, 80:]) / 255.0 * 5.0

    # Внутренний шум (Пейсмейкер)
    noise_left = random.uniform(0.0, 0.06)
    noise_right = random.uniform(0.0, 0.06)

    # Перекрестная логика (Левый глаз управляет правым колесом и наоборот)
    input_left_motor = light_right * 0.4 + motion_right + noise_left
    input_right_motor = light_left * 0.4 + motion_left + noise_right

    left_spike = left_neuron.update(input_left_motor)
    right_spike = right_neuron.update(input_right_motor)

    # --- ПЛАВНАЯ ФИЗИКА ДВИЖЕНИЯ ---
    target_v_left = BOOST_SPEED if left_spike else BASE_SPEED
    target_v_right = BOOST_SPEED if right_spike else BASE_SPEED

    smooth_v_left += (target_v_left - smooth_v_left) * SMOOTHING
    smooth_v_right += (target_v_right - smooth_v_right) * SMOOTHING

    d_angle = (smooth_v_right - smooth_v_left) * 0.05
    fly_angle += d_angle
    
    forward_speed = (smooth_v_left + smooth_v_right) / 2.0
    fly_x += forward_speed * math.cos(fly_angle)
    fly_y += forward_speed * math.sin(fly_angle)

    # Ограничитель экрана (Панический отскок от стен)
    if fly_x < 30 or fly_x > WIDTH - 30 or fly_y < 30 or fly_y > HEIGHT - 30:
        fly_angle += math.pi / 2
        fly_x = max(30, min(WIDTH - 30, fly_x))
        fly_y = max(30, min(HEIGHT - 30, fly_y))

    # --- ОТРИСОВКА ---
    screen.fill((25, 25, 30))
    
    # Муха
    simka.draw.circle(screen, (0, 255, 120), (int(fly_x), int(fly_y)), 16)
    nose_x = fly_x + 22 * math.cos(fly_angle)
    nose_y = fly_y + 22 * math.sin(fly_angle)
    simka.draw.line(screen, (255, 50, 50), (fly_x, fly_y), (nose_x, nose_y), 3)

    # HUD Левый мотор
    simka.draw.rect(screen, (60, 60, 60), (20, 20, 200, 20))
    simka.draw.rect(screen, (0, 200, 255), (20, 20, int(200 * min(1.0, left_neuron.charge / left_neuron.threshold)), 20))
    screen.blit(font.render(f"L Motor Charge: {left_neuron.charge:.2f}", True, (255, 255, 255)), (20, 45))

    # HUD Правый мотор
    simka.draw.rect(screen, (60, 60, 60), (WIDTH - 220, 20, 200, 20))
    simka.draw.rect(screen, (0, 200, 255), (WIDTH - 220, 20, int(200 * min(1.0, right_neuron.charge / right_neuron.threshold)), 20))
    screen.blit(font.render(f"R Motor Charge: {right_neuron.charge:.2f}", True, (255, 255, 255)), (WIDTH - 220, 45))

    simka.display.flip()
    clock.tick(30) # Обработка тактов времени

cap.release()
simka.quit()