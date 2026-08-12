# FlowBelief

Belief-space motion planning experiments using diffusion or flow-matching policies for a custom 2D belief maze environment.

## Installation

Create a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
.
├── belief_env.py          # Custom belief maze environment
├── belief_agent.py        # Belief dynamics and covariance update model
├── belief_API.py          # RRT wrapper and dataset collection entry point
├── Inference.py           # RRT + learned action model inference
├── train.py              # Belief policy training loop
├── train_manager.py      # Configures and launches training
├── cfgs/beliefmaze.yaml  # Training and inference config
├── datasets/             # Belief trajectory dataset
├── metadata/             # Dataset normalization statistics
└── checkpoints/          # Model checkpoints
```

## Usage

Collect belief-space training paths:

```bash
python belief_API.py
```

Train the policy:

```bash
python train_manager.py
```

Run inference:

```bash
python Inference.py
```
