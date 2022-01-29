# Preprocessing Knowledge Graph Dataset

Training IndoKEPLER requires a knowledge graph dataset such as IndoWiki or Wikidata5M. Each can be downloaded from their own respective source. This directory handles the preprocessing of said knowledge graph dataset.

## Initial Dataset

This repository is built to preprocess IndoWiki. However, any datasets with an equivalent format can also be used. Specifically, a datasets must have 2 types of text dataset:
- One entity dataset containing $E$ number of lines, one for each entity in the KG. Each line should contain two strings separated by a tab. The first string should be the unique ID of this entity. The second should be the description of said entity. This usually come from a Wikipedia article with the newlines removed. All descriptions should have no newlines nor tabs.
- Three triplet datasets (One each for training, validation, and testing). Each datasets should contain several lines equal to the number of triplets in it's specific partition. One line contains three strings separated by tabs, describing a triplet.
```
<subject_id> <relation_id> <object_id>
```
`<subject_id>` and `<object_id>` are the ID of subject and object entity respectively. Both IDs should exists in the mentioned entity dataset. `<predicate_id>` should be the ID of the relation between subject and object. For Wikidata, `<predicate_id>` will be an ID starting with letter 'P', such as "P17", "P20", etc.

## Preprocessing Steps

IndoKEPLER preprocessing steps consists of several processes that needed to be run sequentially. Some processes may require a hours to run. Thus, we do not provide an automatic script. The input and output file of each processes are illustrated in the chart below:

### Data Preparation

The first step is to prepare the initial datasets into a more readable form. This involves changing entity and predicate IDs present in the initial dataset to indexes and translating the triplet dataset from text to CSV format.
```python prepare.py <entities_filename> <training_filename> <validation_filename> <test_filename>```
This command will outputs a new entity text file and three triplet CSV files, one for each data split. One row in a CSV file describes one triplet. Specifically, if a row contains a triplet with value '$A$, $B$, $C$', then the subject and object entity of this triplet is the entity described by the ($A+1$)-th and ($B+1$)-th line of the new entity text file respectively.

### Masked Language Model Tokenization

IndoKEPLER requires a MLM dataset for training. Using the new entity text file generated previously, we can create this MLM dataset by running
```python tokenize_mlm.py entities.txt --tokenizer=<tokenizer_name_or_dir> --out-dir=mlm  --num-proc=<number_of_processes>```
This program tokenize the entity descriptions from entities.txt using the specified HuggingFace tokenizer and then transform it into chunks of 512 token sequences. This program should only take several minutes, provided the datasets are small enough. The resulting dataset is saved in the form of a HuggingFace dataset named "mlm" (saved in the directory named "mlm").

### Entity Tokenization

We then create another tokenized dataset, this time for the KE task.
```python tokenize_entities.py entities.txt --tokenizer=<tokenizer_name_or_dir> --num-proc=<number_of_processes>```
This program performs similarly to tokenize_mlm.py without splitting the entity descriptions into chunks. The resulting output is a directory named "tokenized_text" containing the HF dataset.

### True Triplets Generation

Next, we prepare the necessary dataset to perform negative sampling. The first step is to get the list of true heads and tails of each triplet.
```python gen_true_triplets.py --data-dir=csv --out-file-heads=true_heads.txt --out-file-tails=true_tails.txt```

### Data Splitting

Oftentimes, the size of our triplet dataset is too large to be handled all at once. Thus, it may be wise to split it into smaller shards before processing it one-by-one. Although this introduces computation overhead, this allows "checkpointing" for our future operations. Runtime failure on one shard thus will not invalidate the progress that had been done for the other shards.
```python split_dump.py <number_of_splits> --data-dir=csv --out-dir=step-1```
This creates several shards for each partition based on the parameter given. Each data split is sharded separately. Thus, if we give 8 as the number of splits in the program arguments, we will generate 24 shards total, regardless on the number of triplets in each split.

### Negative Sampling

With the true heads and tails collected and the triplets sharded, it is now time to perform negative sampling. The num-proc argument defines the number of threads that will be generated for multiprocessing. Since IndoWiki contains 2.6 million triplets, **A number of 16 processes is recommended**.
```python gen_negative_sampling.py tokenized_text --data-dir=step-1 --out-dir=step-2 --start=<start> --end=<end> --ns=1 --true-heads=true_heads.txt --true-tails=true_tails.txt --num-proc=<number_of_processes>```
Parameter `<start>` and `<end>` allow us to process only a segment of our dataset. For example, suppose that we have sharded our datasets into 8 shards. Setting the start to 5 and end to 7 (one-based) prompts the process to generate negative samples for:
- Training shards with index number 5, 6, and 7
- Validation shards with index number 5, 6, and 7
- Testing shards with index number 5, 6, and 7
Note that each shards are processed separately. There is little to no advantage of running the program once with start 1 and end 8 compared to running the program 8 times for each shard.
As a comparison, running this program on IndoWiki with 16 threads took a total of 6 hours using our hardware (DGX-A100).

### Mapping entities

Next step is to substitute the indexes of each triplet into it's tokenized description. As per (Wang et al. 2020), we take the first 512 tokens of an entity description as representation. **WARNING: 60 GB of space is required when processing IndoWiki. Expect larger values for bigger dataset**.
```python map_entities.py tokenized_text --data-dir=step-2 --start=1 --end=8 --out-dir=ke```
The `<start>` and `<end>` parameter works the same way as in the previous step. unllike negative samping, this process should only takes one hour maximum with IndoWiki.

### MLM and KE Combination

The last step is to create a dataset ready for IndoKEPLER. This means zipping the MLM and KE datasets generated previously together. Both datasets are combined using a "round robin approach". This means that the 1st KE data is paired with the 1st MLM data, the 2nd KE data with 2nd MLM data, and so on. If the number of MLM data is less than the number of KE data (which is almost always the case), the pairing loops back to the first data. So, if the number of MLM data is $M$ and the number of KE data is $N >> M$, then after the $M$-th KE data is paired with the $M$-th MLM data, the $M+1$-th KE data is paired with the 1st MLM data, the $M+2$-th KE data with the 2nd MLM data, and so on.
```python gen_round_robin.py --mlm-dir mlm --ke-dir ke  --out-dir=indokepler```

