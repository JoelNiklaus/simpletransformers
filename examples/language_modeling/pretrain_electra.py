import argparse
import logging
import os

import torch

from prepare_data import prepare_data
from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)

# TODO consider using methods from DeBERTa to improve it
# TODO consider using unigram lm tokenizer: SentencePieceUnigramTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--total_batch_size", type=int, default=256)
    parser.add_argument("--per_device_batch_size", type=int, default=32)
    parser.add_argument("--model_name_or_path", type=str)

    # data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default="")  # default "" for local development
    parser.add_argument("--bucket_name", type=str)
    parser.add_argument("--model_dir", type=str)

    args, _ = parser.parse_known_args()

    os.environ['TOKENIZERS_PARALLELISM'] = "true"

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()

    # IMPORTANT if we set the embedding_size to 128 instead of 768 we get problems if we run the tie_weights() function,
    # the weights of the generator_lm_head (in_features) are changing,
    # leading to dimension errors in matrix multiplication
    # Thus all calls to tie_weights have been disabled. Is this a problem?
    model_args = LanguageModelingArgs(
        # for base version: electra paper says that the generator should be 1/3 of the discriminator's size
        generator_config={
            "max_position_embeddings": 4096,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 4,
        },
        discriminator_config={
            "max_position_embeddings": 4096,
            "embedding_size": 768,
            "hidden_size": 768,
            "num_hidden_layers": 12,
        },
        reprocess_input_data=False,
        overwrite_output_dir=True,
        evaluate_during_training=True,
        n_gpu=num_gpus,  # run with python -m torch.distributed.launch pretrain_electra.py
        num_train_epochs=args.epochs,
        eval_batch_size=args.per_device_batch_size,
        train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=int(args.total_batch_size / args.per_device_batch_size),
        learning_rate=2e-4,  # ELECTRA paper searched in 1e-4, 2e-4, 3e-4, 5e-4
        warmup_steps=10_000,  # as specified in ELECTRA paper
        dataset_type="simple",
        vocab_size=30000,
        block_size=4096,
        use_longformer_electra=True,
        output_dir=os.path.join(args.output_data_dir, "outputs"),
        cache_dir=os.path.join(args.output_data_dir, "cache_dir"),
    )
    data_dir = os.path.join(args.output_data_dir, "data")
    train_file, test_file = data_dir + "/train.txt", data_dir + "/test.txt"

    model = LanguageModelingModel(
        "electra",
        None,
        args=model_args,
        train_files=train_file,
        use_cuda=cuda_available
    )

    # Train the model
    model.train_model(train_file, eval_file=test_file)

    # Evaluate the model
    result = model.eval_model(test_file)