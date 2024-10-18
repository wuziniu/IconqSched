# IconqSched


## Training Iconq concurrent runtime predictor
You can find the saved checkpoints in models/_checkpoints and skip the part on collecting training data and training

### Collecting the training data
In this work we use the IMDB dataset (scaled to 100GB) and the queries used in BRAD paper.

A subset of these queries can be found in workloads/postgres/queries.sql and workloads/redshift/queries.sql. 
We also included the parsed query plans for these queries (parsed_query_plans.json). 
The script parser/parse_plan.py invokes the Postgres/Redshift "EXPLAIN" function and parses the outputs.

Execute the queries with k clients issuing queries in a close loop
```angular2html
mkdir saved_results
python3 run.py \
      --run_k_client_in_parallel \
      --baseline \
      --database postgres \
      --directory workloads/postgres/snowset_1453912639619907921_postgres_replay.csv \
      --save_result_dir saved_results \
      --host 'postgres-imdb.xxxxx.rds.amazonaws.com' \
      --port 5432 \
      --user xxx \
      --password xxx \
      --db_name imdb \
      --query_bank_path workloads/postgres/queries.sql \
      --num_clients $k$ \
      --timeout 1000
```
Vary k with different values. 

### Training Iconq
```angular2html
python3 run.py \
      --train_concurrent_rnn \
      --database postgres \
      --directory saved_results \
      --target_path models/_checkpoints \
      --use_size \
      --use_log \
      --use_table_features \
```



## Testing the IconqSched's scheduling performance

Execute the workload with the DBMS itself (--baseline)
```angular2html
mkdir saved_results
python3 run.py \
      --replay_workload \
      --baseline \
      --database postgres \
      --directory workloads/postgres/snowset_1453912639619907921_postgres_replay.csv \
      --target_path models/_checkpoints \
      --save_result_dir saved_results \
      --host 'postgres-imdb.xxxxx.rds.amazonaws.com' \
      --port 5432 \
      --user xxx \
      --password xxx \
      --db_name imdb \
      --query_bank_path workloads/postgres/queries.sql \
      --timeout 1000
```
Change the --database and --query_bank_path to test on Redshift. 

Execute the workload with the IconqSched
```angular2html
python3 run.py \
      --replay_workload \
      --database postgres \
      --directory workloads/postgres/snowset_1453912639619907921_postgres_replay.csv \
      --target_path models/_checkpoints \
      --save_result_dir saved_results \
      --host 'postgres-imdb.xxxxx.rds.amazonaws.com' \
      --port 5432 \
      --user xxx \
      --password xxx \
      --db_name imdb \
      --query_bank_path workloads/postgres/queries.sql \
      --debug \
      --ignore_short_running \
      --timeout 1000
```

