from box import Box
import json,argparse
from rxngraphormer.train import SPLITRegressorTrainer,SequenceTrainer,SPLITClassifierTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, default='./config/C_H_func_parameters.json')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    para_json = args.config_json
    with open(para_json,'r') as fr:
        config_dict = json.load(fr)
    config = Box(config_dict)
    if config.task == 'regression':
        trainer = SPLITRegressorTrainer(config)
    elif config.task == "sequence_generation":
        config.others.local_rank = args.local_rank
        trainer = SequenceTrainer(config)
    elif config.task == "classification":
        config.others.local_rank = args.local_rank
        trainer = SPLITClassifierTrainer(config)
    else:
        raise NotImplementedError
    trainer.run()
if __name__ == '__main__':
    main()