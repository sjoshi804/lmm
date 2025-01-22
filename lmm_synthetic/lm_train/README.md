# lm-train

## Overview
This repository provides a minimal framework for pre-training GPT-2 language models from scratch using DeepSpeed on custom datasets. Configuration management is handled using Hydra, allowing easy addition of new experiments by specifying overrides in the `config/overrides` folder.

## Features
- **DeepSpeed Integration**: Efficient training with DeepSpeed.
- **Custom Datasets**: Train on your own datasets.
- **Hydra Configuration**: Simplified experiment management with Hydra.

## Getting Started
1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/lm-train.git
    cd lm-train
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run a training script**:
    ```sh
    python train.py
    ```

## License
This project is licensed under the MIT License.

## Contact
For questions or issues, please open an issue on the GitHub repository.
