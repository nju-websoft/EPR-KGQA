# Stardard Libraries
import argparse
import glob
import sys
import time

# Self-defined Modules
from evidence_pattern_retrieval.BERT_Ranker.model_config import get_model_class, load_untrained_model, register_args
from evidence_pattern_retrieval.BERT_Ranker.train_bert_ranker import *
from my_utils.io_utils import read_json, write_json


def predict(args, model, tokenizer, topk=100):
    start = time.time()
    sample_size = args.sample_size

    all_predict_logits = predict_in_batch(args, model, tokenizer)

    split_id = os.path.basename(args.predict_file).split('_')[1]

    prediction_data = read_json(os.path.join(f"data/{args.dataset}/ep_retrieval", 'training_data',
                                             f"{args.dataset}_{split_id}_top{topk}_ap_candi_eps_for_prediction.json"))

    destination_data = integrate_results(prediction_data, all_predict_logits, sample_size, split_id)

    assert len(prediction_data) == len(destination_data)

    print("topk:", topk)
    end = time.time()
    print(f'{split_id} top{topk} EP ranking time', end - start)

    write_json(destination_data, os.path.join(f'data/{args.dataset}/ep_retrieval/', 
                                              f'{args.dataset}_{split_id}_top{topk}_ap_ranked_ep.json'))


def main():
    parser = argparse.ArgumentParser()
    register_args(parser)
    args = parser.parse_args()

    if (os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    # load model for training
    config, tokenizer, model = load_untrained_model(args)
    special_tokens_dict = {'additional_special_tokens': ['[CVT]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = get_model_class(args).from_pretrained(args.output_dir)  # , force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = get_model_class(args).from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    if args.do_predict:
        logger.info("Loading checkpoint %s for prediction", args.model_name_or_path)
        checkpoint = args.model_name_or_path
        logger.info("Do prediction using the following checkpoint: %s", checkpoint)
        model = get_model_class(args).from_pretrained(checkpoint)  # , force_download=True)
        model.to(args.device)
        predict(args, model, tokenizer, topk=args.topk)
    return results


if __name__ == "__main__":
    main()
