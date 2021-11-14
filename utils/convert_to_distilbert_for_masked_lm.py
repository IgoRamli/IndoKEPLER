import argparse
from model.modeling_kepler import KeplerModel, KeplerConfig
from transformers import DistilBertModel, DistilBertForMaskedLM

parser = argparse.ArgumentParser(description="Extract the encoder and MLM head of KEPLER model")
parser.add_argument('--kepler', help="Path to pretrained KEPLER model")
parser.add_argument('--distilbert', help="Destination for the new DistilBert model")

if __name__ == "__main__":
    args = parser.parse_args()
    mlm_head = DistilBertForMaskedLM.from_pretrained('cahya/distilbert-base-indonesian')
    encoder = mlm_head.distilbert
    config = KeplerConfig()
    kepler = KeplerModel.from_pretrained(args.kepler, encoder, mlm_head)
    distilbert_for_masked_lm = kepler.mlm_head
    distilbert_for_masked_lm.save_pretrained(args.distilbert)
