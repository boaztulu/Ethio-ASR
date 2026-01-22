#!/usr/bin/env python
# scripts/train.py

import os
import sys
from pathlib import Path
import argparse
import json
import logging
import random
import numpy as np
import torch
import wandb
from huggingface_hub import login as hf_login


# a function to load environment variables from .env file
def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key] = value.strip("'").strip('"')
                    except ValueError:
                        # Skip lines that don't have the format KEY=VALUE
                        continue
        return True
    else:
        print(f"Warning: .env file not found at {env_path}. "
              "Environment variables will not be loaded.")
        return False

# try to load from .env file in project root
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
env_path = project_root / '.env'
load_env_file(env_path)

# add project root to Python path
sys.path.insert(0, str(project_root))

# Local imports -- don't move these to the top 
from src.utils.config import load_config
from src.data.dataset import (
    load_datasets, 
    build_vocabulary, 
    create_processor, 
    prepare_datasets
)

from src.models.factory import create_asr_model
from src.training.collator import DataCollatorCTCWithPadding
from src.training.trainer import create_asr_trainer

# load huggingface set_seed function from transformers
from transformers import set_seed as huggingface_set_seed

# setup logging
def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

# setup environment variables for wandb and huggingface
def setup_environment():
    """Set up necessary environment variables."""
    # Get API keys from environment (assumes they're already set)
    wandb_key = os.environ.get("WANDB_API_KEY")
    hf_key = os.environ.get("HF_API_KEY")
    
    if not wandb_key:
        logging.warning("WANDB_API_KEY not found in environment variables")
    if not hf_key:
        logging.warning("HF_API_KEY not found in environment variables")

# setup random seeds in numpy, torch, and huggingface for reproducibility
def setup_seed(seed: int):
    """Set up random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # for reproducible operations on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ASR model')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to config YAML file'
    )
    return parser.parse_args()


def main():
    """
    Main function to load the config, setup the environment, 
    parse the arguments, load the configuration, train the model, save the model and processor, 
    evaluate the model, and clean up the dataset cache.
    """
    # setup logging and environment variables
    setup_logging()
    setup_environment()
    
    # parse command line arguments
    args = parse_args()
    
    # load ASR configuration from YAML file
    logging.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # setup random seed for reproducibility
    setup_seed(config.seed)
    huggingface_set_seed(config.seed)

    # login to wandb
    logging.info("Logging in to Weights & Biases...")
    if os.environ.get("WANDB_API_KEY"):
        wandb.login()
        wandb.init(
            project=config.project,
            config={
                'learning_rate': config.learning_rate,
                'model': config.pretrained_model,
                'batch_size': config.batch_size,
                'epochs': config.num_epochs
            }
        )
    else:
        logging.warning("WANDB_API_KEY not found in environment variables. "
                        "Weights & Biases logging will be disabled.")
    
    # log in to huggingface
    logging.info("Logging in to Hugging Face...")
    if os.environ.get("HF_API_KEY"):
        hf_login(token=os.environ["HF_API_KEY"])
        logging.info("Successfully logged in to Hugging Face 🤗")
    else:
        logging.warning("HF_API_KEY not found in environment variables. "
                        "Hugging Face login will be disabled 😔")

    # create output directory 
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # create experiment name and model directory to save the model and processor after training
    experiment_name = config.get_experiment_name()
    model_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # load datasets from the HF hub or local dataset
    logging.info(f"Loading training and evaluation datasets from {config.dataset_path}...")
    train_dataset, eval_dataset = load_datasets(config)
    logging.info(f"Loaded {len(train_dataset)} training samples "
                 f"and {len(eval_dataset)} validation samples")

    # compute the size of the training data as hours
    total_num_of_sec = sum([float(x) for x in train_dataset['audio_duration']])
    num_of_hours = total_num_of_sec/(60*60)
    logging.info(f"Total number of hours: {num_of_hours:.3f}!")

    # build vocabulary from the training dataset
    logging.info("Building vocabulary...")
    
    try:
        language_tags = set(train_dataset['language'])
    except KeyError: 
        # in case no language column is found in the training dataset
        logging.warning("Language tags not found in the training dataset. "
                        "Multilingual model will not be used.")
        language_tags = None
    
    else:
        logging.info(f"Language tags: {language_tags}")

    vocab_dict = build_vocabulary(
        config.character_set,
        config.add_language_tokens,
        language_tags
    )

    logging.info(f"Size of vocabulary created: {len(vocab_dict)}.")
    logging.info(f"Items of the vocabulary: {vocab_dict}...")

    # save vocabulary to file in JSON format to disk
    with open(os.path.join(model_dir, "vocab.json"), "w") as f:
        json.dump(vocab_dict, f, indent=4)
    logging.info(f"Vocabulary saved to {model_dir}/vocab.json")
    
    # create processor and save it to disk
    logging.info("Creating processor...")
    processor = create_processor(config, model_dir)
    logging.info(f"Saving processor to {model_dir}/processor/")
    processor.save_pretrained(model_dir + "/processor/")
    logging.info(f"Type of the processor: {type(processor)}")   

    # create model
    logging.info(f"Creating model from the pretraiend {config.pretrained_model} model...")
    model = create_asr_model(config, processor)
    
    # prepare datasets
    logging.info("Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        train_dataset, eval_dataset, processor
    )

    # show a tokenized sample of the train dataset and deocde text
    logging.info(f"first sample of the train dataset: {train_dataset[0]['labels']}")
    logging.info(f"first sample of the decoded text: {processor.decode(train_dataset[0]['labels'])}")

    # create data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, 
        padding=True
    )
    
    # create and run trainer
    logging.info("Starting training model for ASR...")
    trainer = create_asr_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processor=processor,
        config=config
    )
    
    # train model
    trainer.train()
    
    # save model and processor
    logging.info("Saving model and processor...")
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))

    logging.info(f"Training completed. Model saved to {model_dir}")


    # push model to Hugging Face Hub as a private model
    # if os.environ.get("HF_API_KEY"):
    #     model.push_to_hub(
    #         repo_id=f"badrex/{experiment_name}",
    #         use_auth_token=os.environ["HF_API_KEY"],
    #         private=True
    #     )
    #     logging.info(f"Pushed model to Hugging Face Hub: badrex/{experiment_name}")
    
    # final evaluation
    logging.info("Evaluating model...")
    metrics = trainer.evaluate()
    logging.info(f"Final evaluation metrics: {metrics}")
    
    # save metrics to file in JSON format to disk
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    logging.info(f"Model evaluation completed. Metrics saved to {model_dir}/metrics.json")

    # clean up dataset cache
    logging.info("Cleaning up dataset cache files...")
    train_dataset.cleanup_cache_files()
    eval_dataset.cleanup_cache_files() 

    logging.info("Training & evaluation completed successfully!")


if __name__ == "__main__":
    main()
