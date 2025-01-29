# DeepSeek Django Model Fine-tuning

This project fine-tunes the DeepSeek-R1-1.5b model on Django codebase data for improved Django code generation.

## Features

- Optimized data preprocessing for Django code
- Memory-efficient training process
- Checkpointing and model saving
- Advanced reward function for code quality
- Comprehensive logging

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your Django code is in the `django_data` directory

3. Run training:
```bash
python train.py
```

## Training Process

The training script:
1. Filters and processes Django code files
2. Implements PPO training with custom rewards
3. Saves checkpoints and the best model
4. Uses mixed precision training when GPU is available

## Output

The trained model will be saved in a directory named `fine_tuned_deepseek_django_[timestamp]` with:
- Checkpoints at regular intervals
- Best model based on rewards
- Final model after training
- Training logs
