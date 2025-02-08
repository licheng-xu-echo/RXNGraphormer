from box import Box
import json,argparse
from rxngraphormer.data import MultiRXNDataset,RXNG2SDataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, default='./config/pretrain_parameters.json')
    args = parser.parse_args()
    para_json = args.config_json
    with open(para_json,'r') as fr:
        config_dict = json.load(fr)
    config = Box(config_dict)
    if config.task == "classification":
        rct_dataset = MultiRXNDataset(root=config.data.data_path,name_regrex=config.data.rct_name_regrex,
                                        trunck=config.data.data_trunck,task=config.data.task,file_num_trunck=config.data.file_num_trunck,
                                        name_tag='rct')
        pdt_dataset = MultiRXNDataset(root=config.data.data_path,name_regrex=config.data.pdt_name_regrex,
                                        trunck=config.data.data_trunck,task=config.data.task,file_num_trunck=config.data.file_num_trunck,
                                        name_tag='pdt')
    elif config.task == "sequence_generation":
        train_dataset = RXNG2SDataset(root=config.data.data_path,src_file=config.data.train_src_file,tgt_file=config.data.train_tgt_file,oh=False,multi_process=False,vocab_file=config.data.vocab_file,train=True)
        valid_dataset = RXNG2SDataset(root=config.data.data_path,src_file=config.data.valid_src_file,tgt_file=config.data.valid_tgt_file,oh=False,multi_process=False,vocab_file=config.data.vocab_file,train=False)
        test_dataset = RXNG2SDataset(root=config.data.data_path,src_file=config.data.test_src_file,tgt_file=config.data.test_tgt_file,oh=False,multi_process=False,vocab_file=config.data.vocab_file,train=False)
    else:
        raise ValueError("task must be classification or sequence_generation, regression task is not supported yet")
    
if __name__ == '__main__':
    main()