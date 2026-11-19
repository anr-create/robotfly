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
        
        self.base_speed = 4
        self.speed = self.base_speed
        
        self.evading_ticks = 0
        self.evade_dir = ""
        
        self.front_blocked = False
        self.front_search_dir = ""
        
        self.is_panicking = False
        self.panic_dir = ""

    def move(self):
        if not self.front_blocked:
            self.x += self.speed * math.cos(math.radians(self.angle))
            self.y += self.speed * math.sin(math.radians(self.angle))
            
            if self.x > 500: self.x = 0
            elif self.x < 0: self.x = 500
            if self.y > 500: self.y = 0
            elif self.y < 0: self.y = 500

    def process_hardware_sensors(self):
        if self.front_blocked:
            if self.front_search_dir == "left":
                self.turn_left(10)
            else:
                self.turn_right(10)
                
        elif self.evading_ticks > 0:
            if self.evade_dir == "left":
                self.turn_left(10)
            elif self.evade_dir == "right":
                self.turn_right(10)
            self.evading_ticks -= 1

    def start_side_evasion(self, direction):
        self.evading_ticks = 18
        self.evade_dir = direction

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
            if self.panic_dir == "left":
                self.turn_left(30)
            else:
                self.turn_right(30)
                
            if left_mov < FREE_ROAD and right_mov < FREE_ROAD:
                self.is_panicking = False
                self.speed = self.base_speed  

    def draw(self, screen):
        color = (255, 50, 50) if self.is_panicking else (0, 255, 0)
        
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 20)
        nose_x = self.x + 30 * math.cos(math.radians(self.angle))
        nose_y = self.y + 30 * math.sin(math.radians(self.angle))
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (nose_x, nose_y), 3)

    def turn_left(self, deg=10):
        self.angle -= deg  

    def turn_right(self, deg=10):
        self.angle += deg  


pygame.init()
screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()
fly = Fly(250, 250)

left_brain = Neuron("Left Brain")
right_brain = Neuron("Right Brain")

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

running = True
while running:
    ret, frame = cap.read()
    if not ret: break
    
    height, width, _ = frame.shape
    mid = width // 2
    
    new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    diff = cv2.absdiff(prev_frame_gray, new_frame_gray)
    left_diff = diff[:, 0:mid]
    right_diff = diff[:, mid:width]   
    
    left_movement = cv2.mean(left_diff)[0]
    right_movement = cv2.mean(right_diff)[0]
    
    fly.update_panic_state(left_movement, right_movement)
    
    if not fly.is_panicking and fly.evading_ticks == 0 and not fly.front_blocked:
        if left_movement > 1.0:
            left_brain.add_charge(left_movement)
        if right_movement > 1.0:
            right_brain.add_charge(right_movement)

        if left_brain.charge >= left_brain.threshold:
            fly.turn_left(10)
            left_brain.charge = 0
            
        if right_brain.charge >= right_brain.threshold:
            fly.turn_right(10)
            right_brain.charge = 0
            
    left_brain.update()
    right_brain.update()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False 
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                fly.start_side_evasion("right")
            elif event.key == pygame.K_RIGHT:
                fly.start_side_evasion("left")

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        if not fly.front_blocked:
            fly.front_search_dir = random.choice(["left", "right"])
        fly.front_blocked = True
    else:
        fly.front_blocked = False
        fly.front_search_dir = ""
            
    fly.move()
    fly.process_hardware_sensors()
    
    screen.fill((30, 30, 30))
    fly.draw(screen)
    pygame.display.flip()
    clock.tick(30)
    
    prev_frame_gray = new_frame_gray
    cv2.imshow("Left Eye", left_diff)
    cv2.imshow("Right Eye", right_diff)

cap.release()
cv2.destroyAllWindows()
pygame.quit()