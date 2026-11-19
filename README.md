# RobotFly — Bio-Hybrid LIF Spiking Neuron Obstacle Avoidance

Real-time visual motion detection and obstacle-avoidance prototype inspired by insect compound eyes and Leaky Integrate-and-Fire (LIF) spiking neuron dynamics.

## Project Structure
- `fin_sim_ai.py` / `fin_sim_ai2.py` — main simulation integrating stereo frame differencing (`Left Eye` / `Right Eye`) with LIF-based steering decisions and hardware sensor hooks.
- `final_sim.py` / `simulation.py` — autonomous navigation environment and obstacle dynamics in Pygame.
- `vision_test.py` / `camera_test.py` — optical flow and camera input preprocessing modules.

## Tech Stack
- **Language:** Python 3.14
- **Libraries:** OpenCV (`cv2`), Pygame-CE, NumPy
- **Target Hardware:** Orange Pi 3 LTS / Embedded Linux SBCs