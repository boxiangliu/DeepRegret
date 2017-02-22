# Predicting expression change through joint modeling of regulator and genomic sequences
# Authors
# Boxiang Liu <bliu2@stanford.edu>



#----------- modeling ---------
# Setup: 
mkdir modeling

# Testing a simple model for 2-class classification:
mkdir ../processed_data/initial/
python modeling/initial/run_classification.py --log_dir=../processed_data/initial 

# Testing a model for 3-class classification with dropout:
# Discretization is similar to Midden
mkdir ../processed_data/dropout/
python modeling/dropout/run_classification.py --max_steps=500000 --log_dir=../processed_data/dropout/ &> ../logs/dropout.log
