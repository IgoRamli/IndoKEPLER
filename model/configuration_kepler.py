""" KEPLER Knowledge Embedding model configuration """
from transformers import PretrainedConfig

class KeplerConfig(PretrainedConfig):
    model_type = "kepler"

    def __init__(
        self,
        embedding_size=768,
        vocab_size=30522,
        nrelation=1000,
        gamma=4,
        ke_model='TransE',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.embedding_size = embedding_size  # Must be equal to size of last hidden state
        self.vocab_size = vocab_size  # Must be equal to size of vocabulary
        self.nrelation = nrelation  # Number of distinct relations
        self.gamma = gamma
        self.ke_model = ke_model