pytest --benchmark-columns=ops,rounds,median,mean,stddev,min,max,iterations --benchmark-max-time=5 --benchmark-min-rounds=500 \
       --benchmark-warmup=on --benchmark-warmup-iterations=300 tests/test_performance.py
