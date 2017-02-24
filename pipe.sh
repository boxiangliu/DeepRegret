# Predicting expression change through joint modeling of regulator and genomic sequences
# Authors
# Boxiang Liu <bliu2@stanford.edu>



#----------- modeling ---------
# Setup: 
mkdir modeling evaluation

# Testing a simple model for 2-class classification:
mkdir ../processed_data/initial/
python modeling/initial/run_classification.py --log_dir=../processed_data/initial 

# 3-class classification with dropout,
# Discretization is similar to Middendorf: 
mkdir ../processed_data/dropout/
python modeling/dropout/run_classification.py --max_steps=200000 --log_dir=../processed_data/dropout/ &> ../logs/dropout.log
python evaluation/plot_accuracy.py --train ../processed_data/dropout/train_accuracy.log --val ../processed_data/dropout/val_accuracy.log --test ../processed_data/dropout/test_accuracy.log --fig ../figures/dropout/accuracy.pdf
python modeling/dropout/confusion_matrix.py --graph /srv/persistent/bliu2/deepregret/processed_data/dropout/model.ckpt-199999.meta


# Regression with dropout: 
mkdir ../processed_data/regression/
python -u modeling/regression/run_regression.py --max_steps=200000 --log_dir=../processed_data/regression/ &> ../logs/regression.log
