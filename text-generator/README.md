# Text generator

This project trains a Transformer-based text generator

## Files

- `train.py` : Trains the model and it outputs `training_log.csv` and saves the model to `out/model.pt`

- `generate.py` : Uses the trained model to generate text, and saves `generated_JULIET.txt` and `generated_ROMEO.txt`

- `model.py` : Transformer model definition

- `input.txt` : training datasheet

- `vocab.pkl` : saved vocabulary

- `training_log.csv` : logs training loss 


## Running

1. Train the model: 
```bash
python3 train.py

