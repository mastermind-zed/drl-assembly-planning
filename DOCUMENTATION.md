# DRL Assembly Planning - Project Documentation

This document combines the implementation planning, and verification results for the DRL Assembly Planning project.

---

## 1. Implementation Plan: Long-Term Training (2M Steps)

This plan outlines the steps to execute a 2-million-step training session using the PPO algorithm to match the performance benchmarks in the paper.

### Proposed Changes
I will use the following configurations for the training session:
1. **Steps**: 200 epochs x 10,000 steps per epoch = **2,000,000 total steps**.
2. **Checkpointing**: Enabled via `--save-ckpt` to ensure the final policy is saved to `log/ppo_logs/.../policy.pth`.
3. **WandB Bypass**: Use the `WANDB_MODE=disabled` environment variable to prevent login prompts.
4. **Environment Fix**: The script has been updated to use `DummyVectorEnv` for stability on Windows systems, bypassing issues with multiprocessing hangs.

### Execution Command
```powershell
$env:WANDB_MODE="disabled"; .\.venv\Scripts\python.exe run_ppo_script.py --epoch 200 --step-per-epoch 10000 --save-ckpt --env-id 1 --task-id 1
```

### Verification Plan
- **Log Monitoring**: Check the terminal output to ensure the trainer is progressing.
- **Checkpoint Verification**: Verify that `log/ppo_logs/env1/scene1/seed0/policy.pth` is created.
- **Performance Check**: Run the visualization script with the new checkpoint to verify improved assembly success.

---

## 2. Visualization Script Walkthrough

I have fixed the `demo_visualisation.py` script to correctly load training checkpoints and handle command-line arguments.

### Changes Made
- Updated the script to construct the correct path to training checkpoints: `{ckpt-path}/{algo}_logs/env{env-id}/scene{task-id}/seed{seed}/policy.pth`.
- Added a `--seed` argument (defaults to 0).
- Improved error feedback by printing the target checkpoint path.

### Verification Results
I verified the fix by running the script against an existing checkpoint:
```powershell
.\.venv\Scripts\python.exe demo_visualisation.py --task-id 1 --env-id 1 --algo dqn --ckpt-path log --num-test 1
```
**Output:**
`Loading checkpoint from: log/dqn_logs/env1/scene1/seed0/policy.pth`
`[2026-02-24 22:57:24.566385] Episode: 0, Steps: 100, Rewards: -185.0, Success Components: 1`

### Final Visualization Command
To run the visualization with rendering:
```powershell
.\.venv\Scripts\python.exe demo_visualisation.py --env-id 1 --task-id 1 --algo dqn --ckpt-path log --render
```
