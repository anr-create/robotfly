import pygame
import cv2
import math
import random

class Neuron:
    def __init__(self, name):
        self.name = name
        self.charge = 0
        self.threshold = 3
        self.leak = 0.5

    def update(self):
        self.charge -= self.leak
        if self.charge < 0:
            self.charge = 0
        self.charge += random.uniform(-0.2, 0.2)
    
    def add_charge(self, amount):
        self.charge += amount


class Fly:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0.0
        self.speed = 4
        self.is_chilling = False  # Freeze state flag

    def move(self):
        # Only move if the fly is not chilling/freezing!
        #if not self.is_chilling:
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y += self.speed * math.sin(math.radians(self.angle))
        
        # Screen Wrapping
        if self.x > 500: self.x = 0
        elif self.x < 0: self.x = 500
        if self.y > 500: self.y = 0
        elif self.y < 0: self.y = 500

    def draw(self, screen):
        # Change color to Blue if chilling, Green if active
        color = (0, 150, 255) if self.is_chilling else (0, 255, 0)
        
        pygame.draw.circle(screen, color, (self.x, self.y), 20)
        nose_x = self.x + 30 * math.cos(math.radians(self.angle))
        nose_y = self.y + 30 * math.sin(math.radians(self.angle))
        pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (nose_x, nose_y), 3)

    def turn_left(self):
        self.angle -= 10  # Rotate 10 degrees to the left

    def turn_right(self):
        self.angle += 10  # Rotate 10 degrees to the right 


# 1. Initialize Pygame, screen, and your Fly object
pygame.init()
screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()
fly = Fly(250, 250)

# 2. Initialize your Left and Right Neurons
left_brain = Neuron("Left Brain")
right_brain = Neuron("Right Brain")

# 3. Initialize OpenCV Camera
cap = cv2.VideoCapture(0)

# 4. Grab the very first camera frame for prev_frame_gray
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

running = True
while running:
    # --- A. OPENCV VISION ---
    ret, frame = cap.read()
    height, width, _ = frame.shape
    mid = width // 2
    
    new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    diff = cv2.absdiff(prev_frame_gray, new_frame_gray)
    left_diff = diff[:, 0:mid]
    right_diff = diff[:, mid:width]   
    
    left_movement = cv2.mean(left_diff)[0]
    right_movement = cv2.mean(right_diff)[0]
    
    # --- B. STASIS / CHILL MECHANISM ---
    # Total room movement score
    total_movement = left_movement + right_movement
    
    # If room is completely static (movement < 0.8), freeze/chill!
    """if total_movement < 0.8:
        fly.is_chilling = True
    else:
        fly.is_chilling = False"""

    # --- C. NEURON BRAIN ---
    if left_movement > 1.0:
        left_brain.add_charge(left_movement)
    if right_movement > 1.0:
        right_brain.add_charge(right_movement)

    # --- D. MOTOR OUTPUT ---
    if left_brain.charge >= left_brain.threshold:
        fly.turn_left()
        left_brain.charge = 0
        
    if right_brain.charge >= right_brain.threshold:
        fly.turn_right()
        right_brain.charge = 0
        
    left_brain.update()
    right_brain.update()
    
    # --- E. PYGAME PHYSICS & DRAWING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False 
            
    fly.move()
    screen.fill((30, 30, 30))
    fly.draw(screen)
    pygame.display.flip()
    clock.tick(30)
    
    # --- F. OPENCV UPDATES ---
    prev_frame_gray = new_frame_gray
    cv2.imshow("Left Eye", left_diff)
    cv2.imshow("Right Eye", right_diff)

cap.release()
cv2.destroyAllWindows()
pygame.quit()