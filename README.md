# Midterm Challenge

## Getting Started

```bash
conda create -n dl4ds python=3.10
conda activate dl4ds
python -m pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py --base_config_dir ./config --model_config resnext50_32x4d.yml
```

## Evaluation

```bash
python scripts/evaluate.py --base_config_dir ./config --model_config resnext50_32x4d.yml
```

