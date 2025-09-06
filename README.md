# Flappy Bird RL Mini-Project

This project aims to train & evaluate a PPO agent to play **Flappy Bird**. It makes use of the Gymnasium wrapper developed by [markub3327](https://github.com/markub3327/flappy-bird-gymnasium).

---

## Project Structure

```text
flappybird-ppo-agent/
├── train_flappy.py      # Training script (multi-env PPO with EvalCallback)
├── play_bird.py         # Inference/demo script
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes caches, logs, and models
├── model/               # Saved model checkpoints (timestamped .zip files)
├── logs/                # TensorBoard logs & best-model checkpoints
└── README.md            # This overview
```

---

## Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/Juliane-Farmer/flappybird-ppo-agent.git
   cd flappybird-ppo-agent
   ```
2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Train the agent

```bash
python train_flappy.py
```

### 2. Watch the trained agent play

```bash
python play_bird.py --model-path "./model/ppo_flappy_<timestamp>.zip"
```

> Replace `<timestamp>` with your chosen checkpoint filename.

### 3. Monitor with TensorBoard

```bash
tensorboard --logdir logs/tb/
```

Visit `http://localhost:6006` in your browser.

---

