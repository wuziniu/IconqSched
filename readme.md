## execute Aurora 
first create a folder called "aurora_trace" under the directory where you execute the following script

change line 170 of executor/multi_client_query_execution.py if you want to set a timeout

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

Replay workload simulation
```angular2html
python3 run.py --replay_workload --simulation --directory /Users/ziniuw/Desktop/research/Data/AWS_trace/mixed_postgres --target_path debug/checkpoints --save_result_dir debug/checkpoints 
```

Replay workload real execution with baseline (the DBMS itself)
```angular2html
python3 run.py --replay_workload --baseline --directory /Users/ziniuw/Desktop/research/Data/AWS_trace/mixed_postgres --target_path debug/checkpoints --save_result_dir debug/checkpoints --host 'postgres-imdb.c39astlavjy2.us-east-1.rds.amazonaws.com' --port 5432 --user postgres --password postgres --db_name imdb --database postgres --query_bank_path /Users/ziniuw/Desktop/research/Data/AWS_trace/mixed_postgres/postgres_mixed.sql
```

Replay workload real execution with our scheduler
```angular2html
python3 run.py --ignore_short_running --replay_workload --debug --directory /Users/ziniuw/Desktop/research/Data/AWS_trace/mixed_postgres --target_path debug/checkpoints --save_result_dir debug/checkpoints --host 'postgres-imdb.c39astlavjy2.us-east-1.rds.amazonaws.com' --port 5432 --user postgres --password postgres --db_name imdb --database postgres --query_bank_path /Users/ziniuw/Desktop/research/Data/AWS_trace/mixed_postgres/postgres_mixed.sql --timeout 200
```

Run K clients in parallel with baseline (the DBMS itself)
```angular2html
python3 run.py --run_k_client_in_parallel --debug --target_path debug/checkpoints --save_result_dir debug/checkpoints --host 'imdb-100g.cluster-c39astlavjy2.us-east-1.rds.amazonaws.com' --port 5432 --user postgres --password 'geoffxy_Cirrus22' --db_name imdb_60g --database aurora --query_bank_path /home/ziniuw/data/imdb/workloads/aurora_mixed.sql --timeout 600 --baseline --num_clients 10 --exec_for_s 10000
```

Replay K clients in parallel with our scheduler
```angular2html

```


