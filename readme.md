## execute Aurora 
first create a folder called "aurora_trace" under the directory where you execute the following script

change line 170 of executor/multi_client.py if you want to set a timeout

warm up cluster, this will only start one client 

```angular2html
python3 executor/multi_client.py --schema-name imdb_60g --query-bank-file ~/data/imdb/workloads/aurora_mixed.sql --avg-gap-s 5 --engine aurora --avg-gap-std-s 3 --host 'imdb-100g-primary-00000.xxxx' --port 5432 --user postgres --password 'postgres' --run-warmup --run-warmup-times 4
```

```angular2html
python3 executor/multi_client.py --schema-name imdb_60g --query-bank-file ~/data/imdb/workloads/aurora_mixed.sql --avg-gap-s 5 --engine aurora --num-clients 5 --avg-gap-std-s 3 --host 'imdb-100g-primary-00000.xxxx' --port 5432 --user postgres --password 'postgres'
```

Train LSTM
```angular2html
python3 run.py --use_size --use_log --use_table_features --parsed_queries_path ~/data/concurrency/mixed_aurora/aurora_mixed_parsed_queries.json --directory ~/data/concurrency/mixed_aurora --hidden_size 128 --num_layers 1 --lr 0.01 --loss_function l1_loss --val_on_test
```

