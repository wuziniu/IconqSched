# IconqSched


## Training Iconq concurrent runtime predictor
You can find the saved checkpoints in models/_checkpoints and skip the part on collecting training data and training

### Collecting the training data
In this work we use the IMDB dataset (scaled to 100GB) and the queries used in BRAD paper: https://github.com/mitdbg/brad.
We also used the Cloud Analytic Benchmark based on TPC-H: https://github.com/alexandervanrenen/cab.

A subset of these queries can be found in workloads/postgres/{workload_name}_queries.sql and workloads/redshift/ 
We also included the parsed query plans for these queries ({workload_name}_parsed_query_plans.json). 
Example workload traces are provided in workloads/postgres/{snowset or tpc_sf}_query_trace.csv
For BRAD workload, you can replay other traces using workloads/workload_tools (you can execute python3 run.py --minic_snowset_workload).
For CAB workload, you can modify cab/benchmark-gen to generate different traces.
The script parser/parse_plan.py invokes the Postgres/Redshift "EXPLAIN" function and parses the outputs, you can execute python3 run.py --parse_explain.

utils/load_database.py provides instructions on loading the tpc/imdb data into your Postgres/Redshift clusters.

After loading the tables (which may take a couple hours depending on cluster sizes), you should first warmup your cluster

```angular2html
mkdir saved_results
python3 run.py \
      --warmup_run \
      --database postgres \
      --save_result_dir saved_results \
      --host 'postgres-imdb.xxxxx.rds.amazonaws.com' \
      --port 5432 \
      --user xxx \
      --password xxx \
      --db_name imdb \
      --query_bank_path workloads/postgres/brad_queries.sql \
      --timeout 1000 \
```
Change --database and --query_bank_path to corresponding workload and engines.

After warming up, execute the queries with k clients issuing queries in a close loop
```angular2html
python3 run.py \
      --run_k_client_in_parallel \
      --baseline \
      --database postgres \
      --save_result_dir saved_results \
      --host 'postgres-imdb.xxxxx.rds.amazonaws.com' \
      --port 5432 \
      --user xxx \
      --password xxx \
      --db_name imdb \
      --query_bank_path workloads/postgres/brad_queries.sql \
      --num_clients $k$ \
      --timeout 1000 \
      --scheduler_type None
```
Vary k with different values or set --num_clients_list for a list of clients.
Change --query_bank_path and --db_name for TPC_H.
Change --query_bank_path and --database and the connection strings to for RedShift.

### Training Iconq
```angular2html
python3 run.py \
      --train_concurrent_rnn \
      --model_name postgres_brad \
      --directory saved_results \
      --parsed_queries_path workloads/postgres/brad_parsed_query_plans.sql \
      --target_path models/_checkpoints \
      --rnn_type bilstm \
      --use_size \
      --use_log \
      --use_table_features \
      --ignore_short_running 
```
--directory specifies a directory with all training data files, e.g., the .csv files generated from previous step.
            You can also set --directory to a specific csv file path to be trained only on one file.
Change --model_name to 'redshift_{brad or cab}' and --directory for training a cost model on redshift.


## Testing the IconqSched's scheduling performance

Execute the workload with the DBMS itself (--baseline)
```angular2html
python3 run.py \
      --replay_workload \
      --baseline \
      --database postgres \
      --directory workloads/postgres/snowset_1453912639619907921_postgres_replay.csv \
      --save_result_dir saved_results \
      --host 'postgres-imdb.xxxxx.rds.amazonaws.com' \
      --port 5432 \
      --user xxx \
      --password xxx \
      --db_name imdb \
      --query_bank_path workloads/postgres/brad_queries.sql \
      --timeout 1000
```
Change the --database and --query_bank_path to test on Redshift. 
Change the --target_path, --query_bank_path and --directory to test on TPC-H CAB benchmarks. 

Execute the workload with the IconqSched
```angular2html
python3 run.py \
      --replay_workload \
      --database postgres \
      --model_name postgres_brad \
      --rnn_type bilstm \
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

Change the --scheduler_type to 'lp' or 'qshuffler' to test on baselines. Note that the baselines also need to be trained first.
We tuned the hyper-parameters of IconqSched and other baselines by varying the --steps_into_future, --short_running_threshold, 
--alpha and --starve_penalty on each workload and engine. The numbers reported in the paper are the best tuned results.




