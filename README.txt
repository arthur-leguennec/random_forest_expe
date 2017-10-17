Experience with random forest
By default :
	- 100 trees
	- number of variables in each tree: the squared root of the number of features

For execute an experiment:
	python random_forest.py -dir_out test1 -scales 0.25 0.5 1 -feat_types dim_c2 point_based

This command will create a directory '/expe_test1_dim_c2_point_based/' with a directory named 'dim_c2_point_based__s0_25_s0_5_s1/'.
A confusion matrix no-normalized and normalized will be created too (in png and svg format).
In the same way, a figure of importance features will be also created.
A laz file with the result of classification will be created.