# README for LLaMA Training


## 1. Installing

Before using LLaMa script, ensure that you have the required dependencies installed. You can install them by running the following command:

```bash
pip install -r requirements.txt
```

## 2. Usage


```bash
(base) alecakir@jones-1:~/PlantPal/research/training/llama2$ python llama.py --help
usage: llama.py [-h] -d DATA -o OUTPUT_DIR

This script fine-tune the LLaMa2 with the PlantQA dataset.

options:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Dataset directory
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory

```

## 3. Example

```bash
python llama.py --data ../../data/PlantQA_dataset_8864.csv --output_dir llama-plantqa-out
```

