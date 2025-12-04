# Installation

#### 1. Clone this repository
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT-YT
cd LLaVA-NeXT-YT
```

#### 2. Set up the environment and install dependencies
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install -e .
```

# Training
#### To train the model on the entire dataset, submit the provided SLURM script:

```bash
sbatch slurm_scripts/train_yt_scam.sh
```

# Inference

#### 1. Generate predictions for each video

```bash
sbatch slurm_scripts/eval_yt.sh
```

#### 2. Evaluate accuracy, precision, recall, and F1-score

```python
python llava/eval/calc_results_yt.py
```

# Demo
#### To run inference on a single video, simply execute:

```python
python demo.py
```