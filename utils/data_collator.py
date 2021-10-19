import torch
from transformers.data.data_collator import (
  DataCollatorMixin,
  _torch_collate_batch
)
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass


@dataclass
class DataCollatorForKnowledgeEmbedding(DataCollatorMixin):
  tokenizer: PreTrainedTokenizerBase
  pad_to_multiple_of: int = None
  ns_size: int = 1
  return_tensors: str = "pt"

  def torch_call(self, examples):
    batch_size = len(examples)

    def _extract_encoding(feature_name):
      return [ ke[feature_name] for ke in examples ]

    def _extract_nested_encoding(feature_name):
      return [ [ i for i in ke[feature_name] ] for ke in examples ]

    def _extract_embedding(feature_name):
      return [ ke[feature_name] for ke in examples ]

    def _get_collated_batch(feature_name, extractor):
      ftr = extractor(feature_name)
      ftr = _torch_collate_batch(ftr, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
      return ftr

    heads = _get_collated_batch('heads', _extract_encoding)
    tails = _get_collated_batch('tails', _extract_encoding)
    relations = _extract_embedding('relations')
    heads_r = _get_collated_batch('heads_r', _extract_encoding)
    tails_r = _get_collated_batch('tails_r', _extract_encoding)
    n_heads = _get_collated_batch('nHeads', _extract_nested_encoding)
    n_tails = _get_collated_batch('nTails', _extract_nested_encoding)

    res = {
      'heads': heads,
      'tails': tails,
      'relations': torch.IntTensor(relations),
      'heads_r': heads_r,
      'tails_r': tails_r,
      'nHeads': n_heads.view(batch_size*self.ns_size, -1),
      'nTails': n_tails.view(batch_size*self.ns_size, -1),
      'relations_desc': None,
    }
    return res
