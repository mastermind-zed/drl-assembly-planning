# Walkthrough: Automated ICF Assembly with MARL

This document summarizes the implementation and results for the automated assembly of Insulated Concrete Forms (ICF) using Multi-Agent Reinforcement Learning (MARL).

## Key Accomplishments

### 1. Multi-Agent ICF Environment
- **[NEW] [Construction3DEnv_MARL.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/Construction3DEnv_MARL.py)**: A custom MARL environment based on PettingZoo Parallel API.
- **[NEW] [MultiRobotSite.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/BIMClass/Site/multi_robot_site.py)**: Manages multiple robots, ICF block components, and parallel task allocation.
- **ICF Block Support**: Added specialized ICF component types to `SCO.py`.

### 2. Communication Strategies
Implemented observation masking in the environment to support three key communication models:
- **Centralized**: All agents have full visibility of the site and other agents' positions.
- **Decentralized**: Agents only see their immediate local workspace.
- **Hybrid**: Agents see their local area plus high-level status of other robots.

### 3. MARL Training Algorithms
- **[MAPPO](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/run_mappo_script.py)**: Implemented Multi-Agent Proximal Policy Optimization using Tianshou.
- **[IQL/Q-Learning](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/run_iql_script.py)**: Implemented Independent Q-Learning as a robust value-based baseline.
- **[QMIX Architecture](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/qmix_net.py)**: Implemented the Mixer network using hypernetworks for future joint-Q optimization.

### 4. Verification & Metrics
- **[Metrics Script](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/collect_metrics.py)**: Automatically calculates Average Steps, Success Counts, and Rewards.
- **[Visualization](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/demo_visualisation_marl.py)**: A multi-robot 3D visualization tool for presentation to supervisors.

## Running in VS Code: Step-by-Step

### 1. Open the Project
Open the folder `drl-assembly-planning-main` directly in VS Code.

### 2. Select Python Interpreter
- Press `Ctrl+Shift+P`
- Type `Python: Select Interpreter`
- Choose the environment where you installed the dependencies (likely your base Python or a specific venv).

### 3. Open a Terminal
- Go to `Terminal` -> `New Terminal` in the top menu.
- Ensure you are in the correct directory.

### 4. Run Training (To make robots "learn")
Run the following in the terminal to start a short training session:
```powershell
python run_iql_script.py --epoch 5 --step-per-epoch 100 --num-robots 2
```

### 5. Run Visualization (To see them in action)
To see the robots moving and attempting assembly, run:
```powershell
python demo_visualisation_marl.py --num-robots 2
```

## Will the robots perform the tasks?
**Yes**, but with caveats:
- **Mechanics**: The code contains the full logic for robots to `pick_up` ICF blocks, `move` to targets, and `drop_off` to build walls.
- **Intelligence**: In the initialization phase, robots act randomly. To see them perform the assembly *successfully* and *cooperatively*, they need to be trained for more epochs (e.g., `--epoch 100`).
- **Parallelism**: You will see multiple robots moving at the same time, which is the core of your thesis proposal on multi-robot task allocation.

## Presentation Advice
> [!TIP]
> Use the `demo_visualisation_marl.py` during your presentation. It clearly shows the robots moving in parallel and can be used to explain how different communication strategies (Centralized vs Decentralized) affect their cooperative behavior.
