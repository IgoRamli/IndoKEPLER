# IndoKEPLER Architecture

Implementations of IndoKEPLER can be found in this directory as KeplerModel. KeplerModel consists of a text encoder and two pretraining heads: a Masked Language Modeling head and a Knowledge Embedding head. Therefore, we inherit the DistilBertForMaskedLM class along with it's encoder and MaskedLM head, and then add our own KE head. This model is meant to be used for training phase only. Models trained using our KeplerModel implementation can then be used by loading the model as a DistilBertModel, or a DistilBertForMaskedLM model if one wishes to use the trained MLM head.

## Using Other Text Encoders

It is possible for IndoKEPLER (and KEPLER) to be built on other text encoders such as BERT and RoBERTa (or any language model that can be trained using MLM task and outputs a vector). There are several steps to do this:
1. Create a new class that inherits the MaskedLM model of the new text encoder. For example, to use RoBERTa as the new text encoder, then our new class should inherit RobertaForMaskedLM.
2. Copy our implementation of KE head and modify it to call the encoder appropriately. Since our model uses DistilBertForMaskedLM, it will call self.distilbert to run the encoder. You may want to change this to reflect your new model. Find out how to call your new encoder by diving into the MaskedLM source code.
3. DistilBERT does not accept special tokens mask. You may want to modify this depending on your base text encoder.