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
mkdir -p ../processed_data/regression/ ../figures/regression
python -u modeling/regression/run_regression.py --max_steps=200000 --log_dir=../processed_data/regression/ &> ../logs/regression.log
python evaluation/plot_accuracy.py --train ../processed_data/regression/train_mse.log --val ../processed_data/regression/val_mse.log --test ../processed_data/regression/test_mse.log --fig ../figures/regression/mse.pdf
python modeling/regression/prediction.py --graph /srv/persistent/bliu2/deepregret/processed_data/regression/model.ckpt-199999.meta --log_dir /srv/persistent/bliu2/deepregret/processed_data/regression/
python modeling/regression/plot_prediction.py --pred ../processed_data/regression/prediction.txt --fig ../figures/regression/pred_vs_obs.png


# Perform knockdown experiments using 
# 1) subtracting 3 from regulator expressions. 
# 2) subtracting 10.
# 3) subtracting Inf. 
# 4) to base level of 0. 
python modeling/regression/run_knockdown.py --batch_size 473 --fold_change -3.0 --out ../processed_data/regression/knockdown_prediction.kd_3.txt --graph /srv/persistent/bliu2/deepregret/processed_data/regression/model.ckpt-199999.meta --log_dir /srv/persistent/bliu2/deepregret/processed_data/regression/ 
python modeling/regression/run_knockdown.py --batch_size 473 --fold_change -10.0 --out ../processed_data/regression/knockdown_prediction.kd_10.txt --graph /srv/persistent/bliu2/deepregret/processed_data/regression/model.ckpt-199999.meta --log_dir /srv/persistent/bliu2/deepregret/processed_data/regression/ 
python modeling/regression/run_knockdown.py --batch_size 473 --fold_change 0.0 --out ../processed_data/regression/knockdown_prediction.kd_to_0.txt --graph /srv/persistent/bliu2/deepregret/processed_data/regression/model.ckpt-199999.meta --log_dir /srv/persistent/bliu2/deepregret/processed_data/regression/ 


# Perform overexpression experiment.
python modeling/regression/run_overexpress.py --batch_size 473 --fold_change 3.0 --out ../processed_data/regression/overexpress_prediction.oe_3.txt --graph /srv/persistent/bliu2/deepregret/processed_data/regression/model.ckpt-199999.meta --log_dir /srv/persistent/bliu2/deepregret/processed_data/regression/


# Improving the model with smaller filter size:
mkdir -p ../processed_data/small_filter ../figures/small_filter
python -u modeling/small_filter/run_regression.py --max_steps=200000 --log_dir=../processed_data/small_filter/ &> ../logs/small_filter.log
python evaluation/plot_accuracy.py --train ../processed_data/small_filter/train_mse.log --val ../processed_data/small_filter/val_mse.log --test ../processed_data/small_filter/test_mse.log --fig ../figures/small_filter/mse.pdf
python modeling/regression/prediction.py --graph ../processed_data/small_filter/model.ckpt-199999.meta --log_dir ../processed_data/small_filter/
python modeling/regression/plot_prediction.py --pred ../processed_data/small_filter/prediction.txt --fig ../figures/small_filter/pred_vs_obs.png


# Adding a pooling layer:
mkdir -p ../processed_data/pooling ../figures/pooling
python -u modeling/pooling/run_regression.py --max_steps=200000 --log_dir=../processed_data/pooling/ &> ../logs/pooling.log
python evaluation/plot_accuracy.py --train ../processed_data/pooling/train_mse.log --val ../processed_data/pooling/val_mse.log --test ../processed_data/pooling/test_mse.log --fig ../figures/pooling/mse.pdf
python modeling/regression/prediction.py --graph ../processed_data/pooling/model.ckpt-199999.meta --log_dir ../processed_data/pooling/
python modeling/regression/plot_prediction.py --pred ../processed_data/pooling/prediction.txt --fig ../figures/pooling/pred_vs_obs.png
