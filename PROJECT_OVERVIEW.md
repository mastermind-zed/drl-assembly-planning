# Project Overview: DRL for Assembly Planning

This document provides a simple but comprehensive explanation of what this project does, why it exists, and how the underlying technology works.

---

## 1. The Core Idea: "Brain-Powered Construction"

In traditional robotics, we use **fixed programming**. A human tells a robot: *"Move 5cm right, pick up the block, move 10cm left."* This works in a perfectly clean factory, but it fails on a construction site where:
- Things are moved around (Change in layout).
- Beams get in the way of cranes (Dynamic obstacles).
- Different buildings have different sequences.

**The idea of this project** is to move away from fixed instructions. Instead, we give the robot a **Neural Network (a Brain)** and put it in a simulator. The robot then learns how to build structures by itself through trial and error.

---

## 2. The Reason: Why do we need this?

### A. Mastering Complexity
As a building gets larger, the sequence of assembly becomes a massive puzzle. A human might struggle to find the absolute most efficient path to build 50 columns and beams. The AI can calculate thousands of possibilities per second to find the "perfect" plan.

### B. Real-Time Adaptability
If a shipment of steel is placed in the wrong corner of the yard, a traditional robot would stop because it's "confused." Our model looks at the site, sees the new location, and **instantly re-plans** its path without human intervention.

### C. Safety & Cost
By finding the shortest paths and avoiding all collisions, the system reduces the time a crane is active (saving fuel/electricity) and minimizes the risk of expensive on-site accidents.

---

## 3. How the Model Works (The Logic)

This project uses **Deep Reinforcement Learning (DRL)**. You can visualize this as a game with three main parts:

### The Playground (Environment)
The simulation translates complex engineering data (BIM) into something a computer sees like a 3D grid (voxels). 
- **The Yard**: Where pieces are stored.
- **The Site**: Where pieces need to go.
- **Rules**: We programmed "Civil Engineering Logic" into the playground. For example, a "Beam" cannot stay in the air unless the "Columns" supporting it are already there.

### The Brain (Neural Network - PPO/DQN)
The "Brain" is a mathematical web. It takes in a "Picture" of the 3D grid and decides what to do next.
- **Input**: What the world looks like right now.
- **Action**: One of six moves: *Up, Down, Left, Right, Forward, Backward.*

### The Teacher (Reward Function)
This is the most important part. To make the robot smart, we give it a "Score":
- **Positive Points (+)**: Given when a component reaches its target.
- **Negative Points (-)**: Given for every step taken (to encourage speed) or if the robot hits an obstacle.
- **Final Result**: The robot naturally wants the highest score, so it "learns" that the best way to live is to build the building as fast as possible without hitting anything.

---

## 4. The Training Process: From "Noob" to "Pro"

1.  **Phase 1 (Randomness)**: At the start, the robot has no idea what it's doing. It just moves randomly, hitting walls and wasting time.
2.  **Phase 2 (Discovery)**: After thousands of "games" (Episodes), it accidentally places a column correctly and gets a huge positive reward. It starts to "remember" those moves.
3.  **Phase 3 (Optimization)**: After about 1 to 2 million steps, the robot becomes an "expert." It knows exactly which column to pick first and the tightest path to the target.
4.  **Phase 4 (The Checkpoint)**: This learned expertise is saved as a file: `policy.pth`.

---

## 5. Summary of the Workflow

- **`run_ppo_script.py`**: This is the "School." You run this to teach the robot.
- **`Construction3DEnv_h.py`**: This is the "Simulator." It's the world the robot lives in.
- **`log/`**: This is the "Report Card." It stores the results and the final `policy.pth` brain.
- **`demo_visualisation.py`**: This is the "Presentation." You run this to watch the trained robot actually perform its job in a 3D window.

---

**In short**: This project turns construction into a self-solving puzzle, allowing robots to think, planned, and build autonomously.
