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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per_device_batch_size", type=int,
                        default=64)  # small: 64, base: 16, maxiumum for V100 with 16GB memory
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

    use_longformer_electra = True
    if use_longformer_electra:
        sequence_length = 4096
    else:
        sequence_length = 512

    model = "small"
    generator_divisor = {"small": 4, "base": 3}
    hidden_layers = {"small": 12, "base": 12}
    embedding_size = {"small": 256, "base": 768}  # 128 for small in the original paper
    hidden_size = {"small": 256, "base": 768}
    intermediate_size = {"small": 1024, "base": 3072}
    attention_heads = {"small": 4, "base": 12}
    learning_rate = {"small": 5e-4, "base": 2e-4}  # ELECTRA paper searched in 1e-4, 2e-4, 3e-4, 5e-4
    total_batch_size = {"small": 128, "base": 256}

    # It is easier to just use an existing one, but we may get better performance when training our own later
    train_own_tokenizer = True

    discriminator_config = {
        "max_position_embeddings": sequence_length,
        "num_hidden_layers": hidden_layers[model],
        "embedding_size": embedding_size[model],
        "hidden_size": hidden_size[model],
        "num_attention_heads": attention_heads[model],
        "intermediate_size": intermediate_size[model],
    }
    generator_config = discriminator_config.copy()

    scale_down_num_layers = True
    # Two ways of scaling down the generator
    if scale_down_num_layers:  # just scale down the number of hidden layers
        generator_config['num_hidden_layers'] = generator_config['num_hidden_layers'] // generator_divisor[model]
    else:  # or scale down these three parameters, like described in the paper
        # This doesn't work yet
        # IMPORTANT if we set the embedding_size to 128 instead of 768 we get problems if we run the tie_weights() function,
        # the weights of the generator_lm_head (in_features) are changing,
        # leading to dimension errors in matrix multiplication
        # Thus all calls to tie_weights have been disabled. Is this a problem?
        # generator_config['embedding_size'] = generator_config['embedding_size'] // generator_divisor[model]
        generator_config['hidden_size'] = generator_config['hidden_size'] // generator_divisor[model]
        generator_config['intermediate_size'] = generator_config['intermediate_size'] // generator_divisor[model]
        generator_config['num_attention_heads'] = generator_config['num_attention_heads'] // generator_divisor[model]

    print("Generator", generator_config)
    print("Discriminator", discriminator_config)

    model_args = LanguageModelingArgs(
        # for base version: electra paper says that the generator should be 1/3 of the discriminator's size
        generator_config=generator_config,
        discriminator_config=discriminator_config,
        reprocess_input_data=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=True,
        save_model_every_epoch=False,
        logging_steps=100,
        evaluate_during_training=True,
        evaluate_during_training_silent=False,
        evaluate_during_training_steps=1000,
        evaluate_during_training_verbose=True,
        n_gpu=num_gpus,  # run with python -m torch.distributed.launch pretrain_electra.py
        num_train_epochs=args.epochs,
        eval_batch_size=args.per_device_batch_size * 2,
        train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=total_batch_size[model] // args.per_device_batch_size,
        learning_rate=learning_rate[model],
        warmup_steps=10_000,  # as specified in ELECTRA paper
        dataset_type="simple",
        vocab_size=30000,
        block_size=sequence_length,
        max_seq_length=sequence_length,
        use_longformer_electra=use_longformer_electra,
        tensorboard_dir="tensorboard",
        wandb_project="Longformer-Electra",
        wandb_kwargs={"name": "Electra-Base"},
        output_dir=os.path.join(args.output_data_dir, "outputs"),
        cache_dir=os.path.join(args.output_data_dir, "cache_dir"),
        tokenizer_name="allenai/longformer-base-4096" if not train_own_tokenizer else None,
    )

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    train_file, test_file = data_dir + "/train.txt", data_dir + "/test.txt"

    run_on_sagemaker = False
    if run_on_sagemaker:
        import boto3

        s3 = boto3.resource('s3')
        s3.Bucket(args.bucket_name).download_file(os.path.join(args.model_dir, train_file), train_file)
        s3.Bucket(args.bucket_name).download_file(os.path.join(args.model_dir, test_file), test_file)
    else:
        prepare_data(data_dir, debug=False)

    # TODO run training on additional datasets: posture, eurlex
    model = LanguageModelingModel(
        "electra",
        None,  # "electra", # Add last checkpoint here: TODO save both generator and discriminator
        # generator_name="", # Path to generator checkpoint
        # discriminator_name="", # Path to discriminator checkpoint
        args=model_args,
        train_files=train_file if train_own_tokenizer else None,
        use_cuda=cuda_available
    )

    # Train the model
    model.train_model(train_file, eval_file=test_file)

    # Evaluate the model
    result = model.eval_model(test_file)
