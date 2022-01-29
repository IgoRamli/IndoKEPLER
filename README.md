# IndoKEPLER

This repository is an implementation of IndoKEPLER on HuggingFace framework using DistilBERT as it's base text encoder. It is created to be trained using IndoWiki dataset, although an equivalent *knowledge graph* dataset such as Wikidata5M can also be used.

## Architecture

IndoKEPLER is a BERT based *language model* designed to learn from both textual and factual (knowledge graph) knowledge. It accepts a masked sentences and descriptions of a knowledge graph triplet. IndoKEPLER is t

Our implementation of KEPLER inherits the DistilBert model provided by HuggingFace framework. This allows our model to be easily used for finetuning by simply loading the model using the desired DistilBertModel class.