# Deep Reinforcement Learning for Real-Time Assembly Planning  m

This repository contains the implementation of a self-planning robotic framework for prefabricated construction, as described in the research paper: **"Deep Reinforcement Learning for Real-Time Assembly Planning in Robot-Based Prefabricated Construction"**.

---

## 1. Project Overview: "Brain-Powered Construction"

In traditional robotics, we use **fixed programming**. A human tells a robot: *"Move 5cm right, pick up the block, move 10cm left."* This works in a perfectly clean factory, but it fails on a construction site where:
- Things are moved around (Change in layout).
- Beams get in the way of cranes (Dynamic obstacles).
- Different buildings have different assembly sequences.

**The core idea** is to move away from fixed instructions. Instead, we give the robot a **Deep Neural Network (a Brain)** and put it in a simulator. The robot learns how to build structures by itself through trial and error, adapting to any site configuration.

### Why do we need this?
1.  **Complexity**: As building designs grow, finding the perfect assembly sequence becomes a massive 3D puzzle. AI can solve this faster than any human planner.
2.  **Adaptability**: If the parts are moved to a different part of the yard, the AI **instantly re-plans** its path without needing new code.
3.  **Optimization**: The AI is programmed to find the **shortest possible path**, reducing fuel costs and assembly time.

---

## 2. How the Simulation Works

The simulation is a 3D environment that translates complex engineering data (BIM) into a playable "learning game."

*   **The 3D Grid**: The world is a voxel-based grid (similar to a specialized version of Minecraft). It is divided into two distinct zones:
    *   **The Yard**: The storage area where all raw prefabricated components (SCOs) are initially placed.
    *   **The Construction Area**: The target site where the building must be erected according to the design.
*   **The Components (SCOs)**: The environment primarily handles two types of structural elements:
    *   **Columns**: Vertical supports that form the base of the structure.
    *   **Beams**: Horizontal elements that connect columns.
*   **The Rules (Civil Engineering Logic)**: The simulation isn't just a sandbox; it enforces real-world construction constraints:
    *   **Assembly Sequence**: Columns must be placed and locked before any beams that rely on them can be installed.
    *   **Collision Detection**: The crane cannot move through existing structures or obstacles; it must find a clear path.
*   **The Brain (RL Agent)**: The AI "sees" the world as a 3D tensor and chooses one of **6 possible actions** at every step: *Forward, Backward, Left, Right, Up, or Down*.
*   **The Reward System**: This is the "Teacher" that guides the learning:
    *   **Efficiency Punishment**: The agent loses a small amount of points for every single move. This forces it to find the **shortest possible path**.
    *   **Goal Reward**: The agent receives a large positive "treat" (points) only when a component is successfully and legally placed.
    *   **Completion Bonus**: A final massive reward is given when the entire 13-component structure is finished.

---

## 3. Deep Reinforcement Learning Models

This project implements and compares four distinct DRL "brains" to determine which is best for construction tasks:

### A. DQN (Deep Q-Network)
*   **Concept**: This is a "Value-Based" model. It learns to predict the "Value" (Q-value) of every possible move. 
*   **Best For**: Simple environments (Env 1 & 2). It's great at learning basic paths but can become "confused" as the building grows more complex.

### B. DDQN (Double Deep Q-Network)
*   **Concept**: An improvement over DQN. It uses two neural networks to reduce "overoptimism" (where the robot thinks a bad move is actually good).
*   **Best For**: More stable learning than standard DQN, helping the robot avoid getting stuck in bad habits.

### C. A2C (Advantage Actor-Critic)
*   **Concept**: A "Hybrid" model. It has two parts:
    1.  **The Actor**: Decides which move to make.
    2.  **The Critic**: Evaluates the move and tells the Actor if it was good or bad.
*   **Best For**: Faster learning on machines with multi-core processors, as it can run multiple versions of the game at once.

### D. PPO (Proximal Policy Optimization)
*   **Concept**: The most advanced model in this project. It uses a specialized mathematical "clip" to prevent the brain from changing too drastically at once. This keeps the learning stable and smooth.
*   **Best For**: **Complex Environments (Env 3 & 4)**. PPO is the "gold standard" for this work and is the only model that consistently succeeds in building the largest 13-component structures.

### E. Model Architecture (The "Brain" Structure)
While the learning *algorithms* differ, they all share a similar **Deep Neural Network** architecture (specifically a Multi-Layer Perceptron):

*   **Input (Observations)**: It takes in a tensor representing the 3D voxel grid. Each voxel tells the model:
    *   Is there a component here?
    *   Is this the target location?
    *   Is this voxel part of the yard or construction area?
*   **Hidden Layers**: By default, the brain uses two hidden layers with **256 neurons** each. These layers process the 3D data to find patterns.
*   **Output (Actions)**: It outputs a probability for each of the **6 possible crane actions**:
*     1. Forward   2. Backward   3. Left   4. Right   5. Up   6. Down


---

## 4. Requirements & Setup

### Requirements
- `pygame==2.1.2`, `PyOpenGL==3.1.6`, `tianshou==0.4.9.post1`, `wandb`

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mastermind-zed/drl-assembly-planning.git
    cd drl-assembly-planning
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 5. Training and Visualization

### Basic Training (First Run)
If you want to test if everything is working correctly without waiting for millions of steps, run a short training session (100k steps):
```powershell
# Short verification run (10 epochs x 10k steps)
$env:WANDB_MODE="disabled"; .\.venv\Scripts\python.exe run_ppo_script.py --epoch 100 --step-per-epoch 10000 --save-ckpt --env-id 1 --task-id 1
```

### Long-Term Training (Recommended)
To reach the peak performance shown in the paper (2M steps):
```powershell
$env:WANDB_MODE="disabled"; .\.venv\Scripts\python.exe run_ppo_script.py --epoch 200 --step-per-epoch 10000 --save-ckpt --env-id 1 --task-id 1
```

### Visualizing the Results
Watch your trained model perform the assembly:
```powershell
.\.venv\Scripts\python.exe demo_visualisation.py --env-id 1 --task-id 1 --algo ppo --ckpt-path log --render
```

---

## 6. Analysis & Results
According to the research paper, the **PPO** model achieves the highest success rate because it maintains stability even when the construction site is crowded with obstacles and many components. Use the `plot_tools` to compare the learning curves of different algorithms.


**Paper**: [Deep Reinforcement Learning for Real-Time Assembly Planning in Robot-Based Prefabricated Construction.pdf](Deep_Reinforcement_Learning_for_Real-Time_Assembly_Planning_in_Robot-Based_Prefabricated_Construction.pdf)
