import json,torch,argparse
from box import Box
from rxngraphormer.eval import SeqEval,eval_regression_performance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default="./config/uspto_50k_eval.json")
    args = parser.parse_args()
    with open(args.config_json) as f:
        config_dict = json.load(f)
    config = Box(config_dict)
    if config.task == "sequence_generation":
        seq_eval = SeqEval(trained_model_path=config.trained_model_path,ckpt_file=config.ckpt_file,
                            topk=config.topk,beam_size=config.beam_size,temperature=config.temperature,
                            n_best=config.n_best,min_length=config.min_length,max_length=config.max_length,
                            batch_size=config.batch_size)
        acc = seq_eval.eval(seq_eval.test_dataloader,seq_eval.test_ground_truth_smiles_lst)
    elif config.task == "regression":
        r2,mae,preds,targets = eval_regression_performance(config.trained_model_path,ckpt_file=config.ckpt_file, scale=config.scale,yield_constrain=config.yield_constrain)
        print(f"R2: {r2:.4f}, MAE: {mae:.4f}")
if __name__ == "__main__":
    main()
