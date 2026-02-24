# Implementation Plan: Automated ICF Assembly with MARL

This plan outlines the steps to adapt the `drl-assembly-planning` codebase for the automated assembly of Insulated Concrete Form (ICF) construction using multiple ground robot agents and Multi-Agent Reinforcement Learning (MARL).

## User Review Required

> [!IMPORTANT]
> The transition from single-agent sequential assembly to multi-robot parallel assembly is a significant architectural change. I will be introducing a `MultiAgentConstruction3DEnv` and updating the `BIMClass` to support multiple simultaneous robots.

> [!WARNING]
> I will be using `tianshou`'s MARL capabilities. If there are specific library versions or constraints not mentioned in `requirements.txt`, please let me know.

## Proposed Changes

### [BIMClass & Environment]

#### [MODIFY] [SCO.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/BIMClass/Site/SCO.py)
- Add `ICF` type to the `SCO` class with appropriate dimensions (e.g., standard ICF block sizes).
- Simplify assembly rules for stacking ICF blocks.

#### [MODIFY] [siteOnly_multi_tar.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/BIMClass/Site/siteOnly_multi_tar.py)
- Implement a `Robots` class to manage multiple agents on the site.
- Update `sco_action` or add `multi_sco_action` to handle actions from multiple robots simultaneously.
- Enhance collision detection to include robot-robot and robot-component collisions.

#### [NEW] [Construction3DEnv_MARL.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/Construction3DEnv_MARL.py)
- Create a multi-agent version of the Gym environment.
- Observation space will include local or global views depending on the communication strategy.
- Reward function will be updated to encourage cooperation, task completion, and collision avoidance.

### [RL Algorithms & Training]

#### [NEW] [run_mappo_script.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/run_mappo_script.py)
- Training script for Multi-Agent Proximal Policy Optimization (MAPPO).
- Support for Centralized, Decentralized, and Hybrid communication.

#### [NEW] [run_qmix_script.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/run_qmix_script.py)
- Training script for QMIX (Value-Based Cooperative MARL).

### [Communication Strategies]

#### [NEW] [comm_manager.py](file:///c:/Users/SAMUEL%20ADJEI/Desktop/RL/drl-assembly-planning-main/drl-assembly-planning-main/comm_manager.py)
- Logic to mask/filter observations based on the chosen strategy:
  - **Centralized**: Full state visibility for all agents.
  - **Decentralized**: Agents only see their immediate surroundings.
  - **Hybrid**: Limited sharing of robot positions/status.

## Verification Plan

### Automated Tests
- Run `test_env_marl.py` (to be created) to verify that the multi-agent environment initializes correctly and handles multiple actions.
- Run `run_mappo_script.py --epoch 1` to ensure the training loop is functional.

### Manual Verification
- Visualize the multi-robot ICF assembly using `demo_visualisation_marl.py` (to be created).
- Verify that robots avoid collisions and successfully stack ICF blocks in the simulation.
