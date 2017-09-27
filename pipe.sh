# Predicting expression change through joint modeling of regulator and genomic sequences
# Authors
# Boxiang Liu <bliu2@stanford.edu>



#----------- modeling ---------
#----------- Tensorflow ----------
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
# python -u modeling/regression/run_regression.py --max_steps=200000 --log_dir=../processed_data/regression/ &> ../logs/regression.log
python -u modeling/regression/run_regression.py --max_steps=200000 --log_dir=../processed_data/regression.2/ &> ../logs/regression.2.log
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
/srv/persistent/bliu2/tools/anaconda/bin/python -u modeling/pooling/run_regression.py --max_steps=200000 --log_dir=../processed_data/pooling/ &> ../logs/pooling.log
python evaluation/plot_accuracy.py --train ../processed_data/pooling/train_mse.log --val ../processed_data/pooling/val_mse.log --test ../processed_data/pooling/test_mse.log --fig ../figures/pooling/mse.pdf
python modeling/regression/prediction.py --graph ../processed_data/pooling/model.ckpt-199999.meta --log_dir ../processed_data/pooling/
python modeling/regression/plot_prediction.py --pred ../processed_data/pooling/prediction.txt --fig ../figures/pooling/pred_vs_obs.png


# Use outer product to concatenate: 
mkdir -p ../processed_data/outer_product ../figures/outer_product
python -u modeling/outer_product/run_regression.py --max_steps=200000 --log_dir=../processed_data/outer_product/ &> ../logs/outer_product.log
python evaluation/plot_accuracy.py --train ../processed_data/outer_product/train_mse.log --val ../processed_data/outer_product/val_mse.log --test ../processed_data/outer_product/test_mse.log --fig ../figures/outer_product/mse.pdf
python modeling/regression/prediction.py --graph ../processed_data/outer_product/model.ckpt-199999.meta --log_dir ../processed_data/outer_product/
python modeling/regression/plot_prediction.py --pred ../processed_data/outer_product/prediction.txt --fig ../figures/outer_product/pred_vs_obs.png


#----------- Keras ----------------
# Keras 2:
python modeling/keras2/model.py

# Keras 1:
python modeling/keras1/model.py

# Basset:
python modeling/basset/model.py

# Using smaller filter of size 19 and 11 on the first and second conv layer:
python modeling/keras1_small_filter/model.py
python modeling/keras1_small_filter/interpret.py


# Using tensor product: 
python modeling/single_layer/model.py
python modeling/single_layer/model.reg.py --l1=1e-7 --l2=1e-7 --epochs=60
python modeling/single_layer/model.reg.py --l1=1e-4 --l2=1e-4 --epochs=60
python modeling/single_layer/interpret.py


# Using maxpool after outer product layer:
python modeling/maxpool_outer_product/model.py


# Using LSTM:
python modeling/lstm/model.py
python modeling/lstm/model.concat.py
python modeling/lstm/model.concat.adam.py


# Using GRU:
python modeling/gru/model.concat.py
python modeling/gru/model.concat.pool.100.py
python modeling/gru/model.concat.pool.491.py


# For the CS329M paper: 
python modeling/small_filter/model.simple.py
python modeling/small_filter/model.simple.classification.py

python modeling/single_layer/model.py
python modeling/single_layer/model.classification.py

python modeling/gru/model.concat.pool.100.py
python modeling/gru/model.concat.pool.100.classification.py

python modeling/single_layer/interpret.paper.py


#------------ Concatenation network -------------# 
python modeling/concatenation/concat.class.py
python modeling/concatenation/concat.class.deeplift.py
python modeling/concatenation/concat.regres.py


#------------- Adversarial training ----------# 
python modeling/adversarial/ocncat.regress.adv.py
