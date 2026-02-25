# Deep Reinforcement Learning for Real-Time Assembly Planning

This is the implementation of the framework described in the research paper: **"Deep Reinforcement Learning for Real-Time Assembly Planning in Robot-Based Prefabricated Construction"**.

---

## 1. Requirements & Setup

### Requirements
- `pygame==2.1.2`
- `PyOpenGL==3.1.6`
- `tianshou==0.4.9.post1`
- `wandb`

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mastermind-zed/drl-assembly-planning.git
    cd drl-assembly-planning
    ```

2.  **Install dependencies**:
    Using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Or using Conda:
    ```bash
    conda env create -f environment.yml
    ```

---

## 2. Training Models

The framework supports multiple DRL algorithms: **DQN**, **DDQN**, **A2C**, and **PPO**.

### Basic Training
Use `--env-id` and `--task-id` to select environments (Env1-Env4) and scenarios (S1-S4):
```bash
# Train with PPO (Recommended for complex environments)
python run_ppo_script.py --env-id 1 --task-id 1 --epoch 100 --gamma 0.9 --norm-obs --seed 0
```

### Long-Term Training (2M Steps Benchmarks)
To match the performance benchmarks shown in the research paper, a 2-million-step training session is required. 

> [!IMPORTANT]
> **Windows Users**: The script has been optimized to use `DummyVectorEnv` to avoid multiprocessing hangs common on Windows systems.

**Recommended Execution Command**:
```powershell
# Bypasses WandB login and enables checkpoint saving
$env:WANDB_MODE="disabled"; .\.venv\Scripts\python.exe run_ppo_script.py --epoch 200 --step-per-epoch 10000 --save-ckpt --env-id 1 --task-id 1
```
- **Total Steps**: 2,000,000 (200 epochs x 10k steps)
- **Checkpoints**: Saved automatically to `log/ppo_logs/env{ID}/scene{ID}/seed{ID}/policy.pth`

---

## 3. Visualizing the Trained Agent

Once you have a trained model (`policy.pth`), you can visualize the agent's behavior.

### Fixed Visualization Command
```powershell
.\.venv\Scripts\python.exe demo_visualisation.py --env-id 1 --task-id 1 --algo dqn --ckpt-path log --render
```
- **`--ckpt-path`**: Should point to the `log` folder. The script automatically constructs the path to the specific environment/scenario checkpoint.
- **`--render`**: Enables the 3D PyGame/OpenGL visualization window.

---

## 4. Analysis & Results

### Plotting Training Curves
Log files from the paper are available at the [official releases](https://github.com/hyintell/drl-assembly-planning/releases).
```bash
cd plot_tools
# Download specific data from WandB
python download_data.py --user-name <wandb-username> --group-name Env_1_Scene_1
# Plot results
python extract_info.py --env-id 1 --task-id 1 --plot --logdir <path-to-logs>
```

### Performance Benchmarks
According to the paper, the agent should reach near-optimal success and rewards after ~1M timesteps. For Env1-S1, success components should reach the maximum (columns and beams correctly placed).

---

## 5. Project Information
- **Paper**: [Deep Reinforcement Learning for Real-Time Assembly Planning in Robot-Based Prefabricated Construction.pdf](Deep_Reinforcement_Learning_for_Real-Time_Assembly_Planning_in_Robot-Based_Prefabricated_Construction.pdf)
- **Status**: Research code implementation complete.