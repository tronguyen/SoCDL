#! /bin/bash
root_path=.
num_factors=200
type=ctr-nonopt
i=1
#root_path=../data/arxiv/cv
#num_factors=100

# for i in `seq 1 5`
# do
# #  ./qsub.sh ./ctr --directory $root_path/cv-cf-$i --user $root_path/cf-train-$i-users.dat --item \
# #  $root_path/cf-train-$i-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 0.01 \
# #  --random_seed 33333 --num_factors $num_factors --save_lag 20

#   for type in ofm cf
#   do
#   ./qsub.sh ./ctr --directory $root_path/cv-ctr-$i-$type --user $root_path/$type-train-$i-users.dat --item \
#   $root_path/$type-train-$i-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 \
#   --mult $root_path/mult.dat --theta_init $root_path/theta-vector.dat \
#   --beta_init $root_path/final.beta --num_factors $num_factors --save_lag 20 --theta_opt
#   done

# done


# ./ctr --directory $root_path/cv-ctr-$i-$type --user $root_path/data/folds/$type-train-$i-users.dat --item \
#   $root_path/data/folds/$type-train-$i-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 \
#   --mult $root_path/data/mult.dat --theta_init $root_path/data/theta-vector.dat \
#   --beta_init $root_path/data/beta-vector.dat --num_factors $num_factors --save_lag 20 --theta_opt

# ./ctr --directory $root_path/myctr-$i-$type --user $root_path/mydata/train-users.dat --item \
#   $root_path/mydata/train-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 \
#   --mult $root_path/mydata/mult.dat --theta_init $root_path/mydata/theta-vector.dat \
#   --beta_init $root_path/mydata/beta-vector.dat --num_factors $num_factors

  # ./ctr --directory $root_path/myctr-$i-$type --user $root_path/mydata/train-users.dat --item \
  # $root_path/mydata/train-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 0.01 --num_factors $num_factors

  ./ctr/ctr --mult ./ctr/mydata/mult.dat --theta_init ./trash/lastfm20161017192822/final-theta.dat \
  --beta_init ./trash/lastfm20161017192822/final-beta.dat --directory ./trash/lastfm20161017192822 \
  --user ./data/lastfm/ctr-data/train-users.dat --item ./data/lastfm/ctr-data/train-items.dat --max_iter 1 \
  --num_factors 200 --lambda_v 100.000000 --lambda_u 0.100000 --save_lag 100 --random_seed 123 --delta_init \
  ./trash/lastfm20161017192822/final-delta.dat