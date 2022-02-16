# IndoKEPLER

This repository contains the implementation of IndoKEPLER on HuggingFace framework using DistilBERT as it's base text encoder. It is created to be trained using ![IndoWiki](https://github.com/IgoRamli/IndoWiki) dataset, although an equivalent *knowledge graph* datasets like ![Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m) can also be used.

## Architecture

IndoKEPLER is a BERT based *language model* designed to learn from both textual and factual (knowledge graph) knowledge. It accepts a masked sentences and descriptions of a knowledge graph triplet. Our implementation of KEPLER inherits the DistilBERT model provided by HuggingFace framework. This allows our model to be easily used for finetuning by simply loading the model using the desired DistilBertModel class.

## Reference

IndoKEPLER is part of my undergraduate thesis that can be accessed ![here](https://drive.google.com/file/d/1dLZMEnoIDlsBPd3XyyPkz7TJOgl0AALN/view?usp=sharing).

An english paper of the work can be accessed here (WIP).