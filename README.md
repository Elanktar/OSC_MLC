# Github repository and supplementary material for the article "Operational Evaluation of Algorithms for Online Streaming Continual Multi-Label Classification".
___

## Data :
- Clusters : contains tasks created from different data sets
- Eval_set : contains evaluation sets of the different tasks
- Experiences : contains the learning experiences of the different tasks
- Labels : contains the label signature of the tasks
- Length : contains the length of the learning experiences
- Orders : contains task orders for each stream
- datasets : the dataset Mediamill is too heavy to be found in the repository, but can be found at https://www.uco.es/kdis/mllresources/

___

## Results :
- bench_metrics : contains the code of the metrics not provided by River
- consumption : consumption measures of the algorithms (with tables)
- graphs : contains all result graphs
- results : contains results and tables

___

## Models :
- Config : contains the configurations with hyperparameters selected after HPO
- implemented_models : contains implemented model

___

## Scripts :
For each script, write "bash name_file.sh" in the linux terminal.
The parameter of the associated python file can be changed in the .sh file.
The "parameters.py" file must be updated with the algorithms that will be tested, and the datasets with number of features and labels.

- 1_task_generator : generates tasks from a tabular multi-label dataset (.arff)
- 2_order_generator : generates a task order for the data stream
- 3_benchmark : execute HPO, then evaluate the model on a data stream
- 4_graph_generator : generates the result figures for online metrics
- 5_CL_eval_process : generates the result figures for post-experience evaluation metrics
- 6_table_generator : generates result tables and execute statistical tests
- 7_speed_bench : execute speed learning evaluation
