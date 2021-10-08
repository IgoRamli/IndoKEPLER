from transformers import HfArgumentParser

from util.functions import prepare_trainer

parser = HfArgumentParser(description='Trains a pretrained model using KEPLER method')
parser.add_argument('--mlm-dirs', required=True, help='Path to MLM dataset directories, separated by commas (,)')
parser.add_argument('--ke-dirs', required=True, help='Path to KE dataset directories, separated by commas (,)')

if __name__ == '__main__':
    args = parser.parse_args()
    trainer = prepare_trainer(args)
    trainer.train(resume_from_checkpoint=True)