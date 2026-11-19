import cv2


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
    

    def add_charge(self, amount):
        self.charge += amount
        if self.charge >= self.threshold:
            print(f"{self.name} Motor is working!")
            self.charge = 0

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read() 
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
left_brain = Neuron("Left Brain")
right_brain = Neuron("Right Brain")
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    mid = width // 2
    new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    diff = cv2.absdiff(prev_frame_gray, new_frame_gray)
    left_diff = diff[:, 0:mid]
    right_diff = diff[:, mid:width]
    left_movement = cv2.mean(left_diff)[0]
    right_movement = cv2.mean(right_diff)[0]
    if left_movement > 1.0:
        left_brain.add_charge(left_movement)
    if right_movement > 1.0:
        right_brain.add_charge(right_movement)
    cv2.imshow("Left Eye", left_diff)
    cv2.imshow("Right Eye", right_diff)
    left_brain.update()
    right_brain.update()
    prev_frame_gray = new_frame_gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()