from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers import DistilBertForMaskedLM, logging
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import gelu
from .configuration_kepler import KeplerConfig

logger = logging.get_logger(__name__)

class KeplerForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    ke_loss: Optional[torch.FloatTensor] = None
    pScore: torch.FloatTensor = None
    nScore: torch.FloatTensor = None


class KeplerModel(DistilBertForMaskedLM):
    config_class = KeplerConfig
    base_model_prefix = "kepler"

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # KE configuration
        self.nrelation = config.nrelation
        self.gamma = nn.Parameter(
            torch.Tensor([config.gamma]),
            requires_grad = False
        )
        self.eps = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.eps) / config.dim]),
            requires_grad = False
        )
        self.relation_embedding = nn.Embedding(config.nrelation, config.dim)
        nn.init.uniform_(
            tensor = self.relation_embedding.weight,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item()
        )

        model_func = {
            'TransE': self.TransE,
        }
        self.score_function = model_func[config.ke_model]

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=2, dim=2)
        return score

    def compute_ke_score(self, heads, tails, nHeads, nTails, heads_r, tails_r, relations, relations_desc_emb=None):
        heads = heads[:, 0, :].unsqueeze(1)
        tails = tails[:, 0, :].unsqueeze(1)
        heads_r = heads_r[:, 0, :].unsqueeze(1)
        tails_r = tails_r[:, 0, :].unsqueeze(1)

        nHeads = nHeads[:, 0, :].view(heads.size(0), -1, self.config.dim)
        nTails = nTails[:, 0, :].view(tails.size(0), -1, self.config.dim)

        if relations_desc_emb is not None:
            relations = relations_desc_emb[:, 0, :].unsqueeze(1)
        else:
            relations = self.relation_embedding(relations).unsqueeze(1)

        heads = heads.type(torch.cuda.FloatTensor)
        tails = tails.type(torch.cuda.FloatTensor)
        nHeads = nHeads.type(torch.cuda.FloatTensor)
        nTails = nTails.type(torch.cuda.FloatTensor)
        heads_r = heads_r.type(torch.cuda.FloatTensor)
        tails_r = tails_r.type(torch.cuda.FloatTensor)

        relations = relations.type(torch.cuda.FloatTensor)

        pScores = (self.score_function(heads_r, relations, tails) + self.score_function(heads, relations, tails_r)) / 2.0
        nHScores = self.score_function(nHeads, relations, tails_r)
        nTScores = self.score_function(heads_r, relations, nTails)
        nScores = torch.cat((nHScores, nTScores), dim=1)
        return pScores, nScores

    def ke_forward(self, heads, tails, nHeads, nTails, heads_r, tails_r, relations, relations_desc, relation_desc_emb=None, **kwargs):
        if relations_desc is not None:
            relation_desc_emb, _ = self.distilbert(relations)  # Relations is encoded
        else:
            relation_desc_emb = None # Relation is embedded

        ke_states = {
            'heads': self.distilbert(heads)[0],
            'tails': self.distilbert(tails)[0],
            'nHeads': self.distilbert(nHeads)[0],
            'nTails': self.distilbert(nTails)[0],
            'heads_r': self.distilbert(heads_r)[0],
            'tails_r': self.distilbert(tails_r)[0],
            'relations': relations,
            'relations_desc_emb': relation_desc_emb,
        }

        pScores, nScores = self.compute_ke_score(**ke_states)

        pLoss = F.logsigmoid(pScores).squeeze(dim=1)
        nLoss = F.logsigmoid(-nScores).mean(dim=1)
        ke_loss = (-pLoss.mean()-nLoss.mean())/2.0
        return pScores, nScores, ke_loss

    def mlm_forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )

    def forward(
        self,
        mlm,
        heads,
        tails,
        nHeads,
        nTails,
        heads_r,
        tails_r,
        relations,
        relations_desc_emb=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mlm_output = self.mlm_forward(**mlm)

        pScore, nScore, ke_loss = self.ke_forward(heads,
                                                  tails,
                                                  nHeads,
                                                  nTails,
                                                  heads_r,
                                                  tails_r,
                                                  relations,
                                                  relations_desc=None)

        loss = None
        if ke_loss is not None and mlm_output.loss is not None:
            loss = mlm_output.loss + ke_loss

        if not return_dict:
            output = (mlm_output.loss, ke_loss, pScore, nScore)
            return ((loss,) + output) if loss is not None else output

        return KeplerForPreTrainingOutput(
            loss=loss,
            mlm_loss=mlm_output.loss,
            ke_loss=ke_loss,
            pScore=pScore,
            nScore=nScore,
        )