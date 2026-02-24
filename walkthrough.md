# Visualization Script Walkthrough

I have fixed the `demo_visualisation.py` script and determined the correct command for you to run.

## Changes Made

### [DQN Visualization Component]

#### [demo_visualisation.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning/demo_visualisation.py)
- Updated the script to correctly construct the path to your training checkpoints: `{ckpt-path}/{algo}_logs/env{env-id}/scene{task-id}/seed{seed}/policy.pth`.
- Added a `--seed` argument (defaults to 0).
- Improved error feedback by printing the exact path it attempts to load.

## Verification Results

I verified the fix by running the script against your existing checkpoint at `log\dqn_logs\env1\scene1\seed0\policy.pth`:

```powershell
.\.venv\Scripts\python.exe demo_visualisation.py --task-id 1 --env-id 1 --algo dqn --ckpt-path log --num-test 1
```

**Output:**
```
Loading checkpoint from: log/dqn_logs/env1/scene1/seed0/policy.pth
[2026-02-24 22:57:24.566385] Episode: 0, Steps: 100, Rewards: -185.0, Success Components: 1
```

## Final Command

To run the visualization with rendering enabled, use the following command:

```powershell
.\.venv\Scripts\python.exe demo_visualisation.py --env-id 1 --task-id 1 --algo dqn --ckpt-path log --render
```

> [!NOTE]
> Ensure that you use `--task-id 1` instead of `--test-id 1` as previously tried, as the script expects `task-id`.
