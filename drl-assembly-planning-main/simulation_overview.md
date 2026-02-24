# Technical Report: Automated Multi-Robot ICF Assembly with MARL

## 1. Project Overview
This project simulates the automated assembly of Insulated Concrete Forms (ICFs) using multiple ground robots. The core objective is to move from a single-agent system to a scalable **Multi-Agent Reinforcement Learning (MARL)** framework that supports collaborative task allocation and efficient construction.

## 2. Technical Architecture

### 2.1 Multi-Agent Environment
The foundation is the **`Construction3DEnvMARL`** class, built using the **PettingZoo Parallel API**. 
- **Parallel Execution**: Actions for all robots are processed simultaneously in each timestep.
- **State Space**: Includes robot position, battery status, carried component info, and relative distances to other agents.
- **Action Space**: Discrete actions (Forward, Backward, Left, Right, Up, Down, PickUp, Place).

### 2.2 Communication Strategies
To research how robot "collaboration" affects efficiency, three communication models were implemented:
- **Centralized**: Global site knowledge; robots know precisely where everyone is.
- **Decentralized**: Local sensing; robots only "see" others within a specific visibility radius.
- **Hybrid**: A mix of local high-resolution data and global status summaries.

### 2.3 RL Algorithms
Two state-of-the-art MARL approaches are supported:
1. **MAPPO (Multi-Agent PPO)**: An on-policy actor-critic method that uses a centralized critic for better stability during multi-agent training.
2. **IQL (Independent Q-Learning)**: A robust baseline where each agent learns its own Q-function, using shared weights for faster convergence.

---

## 3. Installation Guide

### Prerequisites
- Python 3.11.x
- VS Code

### Step-by-Step Setup
1. **Open Project**: Open the `drl-assembly-planning-main` folder in VS Code.
2. **Create Environment**:
   ```powershell
   python -m venv .venv
   ```
3. **Set Execution Policy** (If on Windows PowerShell):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```
4. **Activate Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
5. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 4. How to Use the Simulation

### A. Training the AI
Run this to make the robots "learn" how to build. For the first test, use low epochs:
```powershell
python run_iql_script.py --epoch 5 --num-robots 2
```
*Increase `--epoch 100` for more intelligent behavior.*

### B. Visualizing Results (3D Presentation)
Use this tool to show the simulation in 3D. It includes camera controls:
```powershell
python demo_visualisation_marl.py --num-robots 2
```
- **Arrows**: Rotate camera.
- **W/S Keys**: Zoom in and out.

### C. Collecting Metrics
Generate performance stats (Steps, Successful placement, etc.):
```powershell
python collect_metrics.py --episodes 5 --strategy centralized
```

---

## 5. Implementation Process (Success Roadmap)
The project followed a rigorous 4-phase transformation:
1. **Environmental Refactoring**: Converted the existing single-agent `BIMClass` into a `MultiRobotSite` capable of handling parallel agents and ICF-specific components.
2. **MARL Integration**: Wrapped the site into a PettingZoo-compatible Gym environment and resolved complex batching issues between `Tianshou` and `Torch`.
3. **Communication Logic**: Implemented observation masking to simulate varying levels of inter-robot communication.
4. **Verification**: Validated the system using both a custom 3D OpenGL visualizer and automated metrics collection.

## 6. Conclusion
The resulting system provides a robust sandbox for researching multi-robot construction. By adjusting the number of robots and communication strategies, one can analyze the trade-offs between coordination overhead and construction speed, directly supporting the core thesis proposal objectives.
