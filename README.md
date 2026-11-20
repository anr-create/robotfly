# RobotFly — Bio-Hybrid LIF Spiking Neuron Obstacle Avoidance

Real-time visual motion detection and obstacle-avoidance prototype inspired by insect compound eyes and **Leaky Integrate-and-Fire (LIF)** spiking neuron dynamics. Built for low-compute embedded edge hardware without heavy deep-learning models.

## How It Works
- **Stereo Visual Input (Left / Right Eye):** Splits visual frames into left and right sensory fields, using frame differencing (`cv2.absdiff`) to extract relative motion and looming cues.
- **Spiking Neuron Integration (LIF):** Converts visual motion intensity into membrane potential accumulation with leak decay, firing steering spikes when obstacle proximity thresholds are crossed.
- **Closed-Loop Avoidance Simulation:** Runs the full perception-to-action loop inside a real-time 2D Pygame obstacle environment with hardware sensor hooks ready for embedded deployment.

## Project Structure
- `fin_sim_ai.py` — complete self-contained simulation integrating stereo frame differencing (Left Eye / Right Eye), LIF spiking neuron steering logic, Pygame 2D obstacle environment, and hardware sensor hooks.

## Tech Stack
- **Language:** Python 3.14
- **Libraries:** OpenCV (`cv2`), Pygame-CE, NumPy
- **Target Hardware:** Orange Pi 3 LTS / Embedded Linux SBCs

## Quick Start
```bash
pip install opencv-python pygame-ce numpy
python fin_sim_ai.py
