# Long-Term Training Session (2M Steps)

This plan outlines the steps to execute a 2-million-step training session using the PPO algorithm to match the performance benchmarks in the paper.

## Proposed Changes

### [Training Setup]

I will provide a command that runs `run_ppo_script.py` with the following configurations:
1.  **Steps**: 200 epochs x 10,000 steps per epoch = **2,000,000 total steps**.
2.  **Checkpointing**: Enabled via `--save-ckpt` to ensure the final policy is saved to `log/ppo_logs/.../policy.pth`.
3.  **WandB Bypass**: Use the `WANDB_MODE=disabled` environment variable to prevent login prompts from blocking execution.

## Execution Command

```powershell
$env:WANDB_MODE="disabled"; .\.venv\Scripts\python.exe run_ppo_script.py --epoch 200 --step-per-epoch 10000 --save-ckpt --env-id 1 --task-id 1
```

## Verification Plan

### Manual Verification
- **Log Monitoring**: Check the terminal output to ensure the trainer is progressing through epochs.
- **Checkpoint Verification**: Verify that `log/ppo_logs/env1/scene1/seed0/policy.pth` is created upon completion.
- **Performance Check**: After training, run the visualization script with the new checkpoint to verify improved assembly success.
