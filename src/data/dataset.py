# src/data/dataset.py
import json
import logging
from typing import Dict, Tuple, List, Any, Optional
from datasets import load_dataset, Dataset, Audio, DatasetDict

from datasets import disable_caching
disable_caching()

from tqdm import tqdm

from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertProcessor,
    Wav2Vec2Processor
)

from src.data.preprocessing import (
    clean_text_batch, 
    prepare_dataset_batch
)

from src.utils.config import ASRConfig

# type alias for processor
ASRProcessor = Wav2Vec2Processor | Wav2Vec2BertProcessor


def load_datasets(config: ASRConfig) -> Tuple[Dataset, Dataset]:
    """Load and prepare datasets for training and evaluation.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load custom dataset if specified
    if hasattr(config, 'use_custom_dataset') and config.use_custom_dataset:
        if hasattr(config, 'dataset_path') and config.dataset_path:
            logging.info(f"Loading custom training dataset locally from "
                         f"{config.dataset_path}...")
            
            dataset = DatasetDict.load_from_disk(config.dataset_path)
        else:
            raise ValueError(f"dataset_path to a local dataset must be specified "
                             f"when use_custom_dataset is True")
    else:
        # Load dataset from HF hub
        logging.info(f"Loading training dataset from HF hub from "
                     f"{config.dataset_path}...")
        dataset = load_dataset(
            config.dataset_path,
            config.language,  # only for the Ethiopian speech corpus
            verification_mode="no_checks", 
        )
    
    # making splits
    logging.info(f"Creating train and validations splits...")
    train_dataset = dataset[config.train_split]
    dev_dataset = dataset[config.eval_split]


    # cast audio column to Audio with 16000 Hz sampling rate
    logging.info(f"Casting audio column to Audio with 16000 Hz sampling rate...")
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    dev_dataset = dev_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # if there is a feature called "transcript" rename it to "transcription"
    # if it is called "transcription" just keep it
    if "transcription" in train_dataset.column_names:
        pass
    elif "transcript" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("transcript", "transcription")
    # else if it is "text" rename it to "transcription"
    elif "text" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("text", "transcription")    
    else:
        raise ValueError(f"Transcription column was not found in train dataset,"
                         f"which should be called 'transcript', 'text', or 'transcription'."
                         f"Found columns: {train_dataset.column_names}.")
    
    # same for dev dataset
    if "transcription" in dev_dataset.column_names:
        pass
    elif "transcript" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("transcript", "transcription")
    elif "text" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("text", "transcription")
    else:
        raise ValueError(f"Transcription column was not found in validation dataset,"
                         f"which should be called 'transcript', 'text', or 'transcription'."
                         f"Found columns: {dev_dataset.column_names}.")

    # if there is a column called "duration" rename it to "audio_duration"
    if "duration" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("duration", "audio_duration")
    elif "audio_duration" in train_dataset.column_names:
        pass
    else:
        # create audio_duration column
        audio_duration_list = []

        for audio in tqdm(train_dataset["audio"], 
                               total=len(train_dataset["audio"]),  # Use multiple CPU cores for parallel processing
                               desc="Calculating audio duration in train dataset"):
            try:
                audio_duration_list.append(len(audio["array"]) / audio["sampling_rate"])
            except Exception as e:
                logging.error(f"Error calculating audio duration for audio {audio}: {e}")
                audio_duration_list.append(0.0)

        logging.info(f"Creating audio_duration column in train dataset...")
        train_dataset = train_dataset.add_column(
            "audio_duration", audio_duration_list
        )

    # sample dataset if specified
    if config.sample:
        logging.info(f"Sampling dataset to {config.sample_size} samples...")

        # shuffle the train dataset
        train_dataset = train_dataset.shuffle(seed=config.seed)
        # select the first config.sample_size samples
        train_dataset = train_dataset.select(range(config.sample_size))

        # shuffle the dev dataset
        dev_dataset = dev_dataset.shuffle(seed=config.seed)
        # select the first config.sample_size samples 
        # => changed to 2000 sampels because valdiation set is large
        dev_dataset = dev_dataset.select(range(2000))

    # same for dev dataset
    if "duration" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("duration", "audio_duration")
    elif "audio_duration" in dev_dataset.column_names:
        pass
    else:
        # create audio_duration column
        audio_duration_list = []

        for audio in tqdm(dev_dataset["audio"], 
                               total=len(dev_dataset["audio"]), 
                               desc="Calculating audio duration in validation dataset"):
            
            try:
                audio_duration_list.append(len(audio["array"]) / audio["sampling_rate"])
            except Exception as e:
                logging.error(f"Error calculating audio duration for audio {audio}: {e}")
                audio_duration_list.append(0.0)

        logging.info(f"Creating audio_duration column in validation dataset...")
        dev_dataset = dev_dataset.add_column(
            "audio_duration", audio_duration_list
        )


    # if there is a column called "audio_filepath" rename it to "audio"
    if "audio_filepath" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("audio_filepath", "audio")
    elif "audio" in train_dataset.column_names:
        pass
    else:
        raise ValueError(f"Audio filepath column was not found in train dataset,"
                         f"which should be called 'audio_filepath' or 'audio'."
                         f"Found columns: {train_dataset.column_names}.")
    
    # same for dev dataset  
    if "audio_filepath" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("audio_filepath", "audio")
    elif "audio" in dev_dataset.column_names:
        pass
    else:
        raise ValueError(f"Audio filepath column was not found in validation dataset,"
                         f"which should be called 'audio_filepath' or 'audio'."
                         f"Found columns: {dev_dataset.column_names}.")
    
    # Remove features not used in training 
    logging.info(f"Removing unnecessary columns...")
    features_to_keep = [
        "audio", "transcription", "audio_duration", "language",
    ]

    features_to_remove = [f for f in train_dataset.features if f not in features_to_keep]

    train_dataset = train_dataset.remove_columns(features_to_remove)
    dev_dataset = dev_dataset.remove_columns(features_to_remove)
    
    # # remove samples that are longer than a max duration threshold
    # max_duration = 42.0 
    # logging.info(f"Removing samples that are longer than {max_duration} seconds...")
    # train_dataset = train_dataset.filter(
    #     lambda x: x["audio_duration"] < max_duration,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing long samples in train split"
    # )

    # dev_dataset = dev_dataset.filter(
    #     lambda x: x["audio_duration"] < max_duration,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing long samples in validation split"
    # )

    # # remove samples that are shorter than one second
    # logging.info(f"Removing samples that are shorter than one second...")
    # train_dataset = train_dataset.filter(
    #     lambda x: x["audio_duration"] > 1.0,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing short samples in train split"
    # )
    # dev_dataset = dev_dataset.filter(
    #     lambda x: x["audio_duration"] > 1.0,
    #     num_proc=4,  # Use multiple CPU cores for parallel processing
    #     desc="Removing short samples in validation split"
    # )

    # Preprocess text transcripts by removing special characters
    logging.info(f"Preprocessing text transcripts...")
    train_dataset = train_dataset.map(
        lambda batch: clean_text_batch(batch, config.character_set, config.apply_accent_replacements),
        batched=True,
        batch_size=64,
        # disable caching
        #keep_in_memory=True, 
        #load_from_cache_file=False,
        desc="Cleaning text transcripts in train split"
    )
    dev_dataset = dev_dataset.map(
        lambda batch: clean_text_batch(batch, config.character_set, config.apply_accent_replacements),
        batched=True,
        batch_size=64,
        # disable caching
        #keep_in_memory=True,
        #load_from_cache_file=False,
        desc="Cleaning text transcripts in validation split"
    )

    # add language tokens to the beginning of the transcription
    if config.add_language_tokens:        
        train_dataset = train_dataset.map(
            lambda batch: add_language_tag_to_transcript(batch),
            batched=True,
            batch_size=64,
            # disable caching
            #keep_in_memory=True,
            #load_from_cache_file=False,
            desc="Adding language tags to train transcriptions"
        )

        dev_dataset = dev_dataset.map(
            lambda batch: add_language_tag_to_transcript(batch),
            batched=True,
            batch_size=64,
            # disable caching
            #keep_in_memory=True,
            #load_from_cache_file=False,
            desc="Adding language tags to validation transcriptions"
        )

    return train_dataset, dev_dataset


def add_language_tag_to_transcript(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Add language tags to the beginning of the transcription"""

    # word delimiter "|" was used instead of space after language tag
    # tokenizer strips whitespace around special tags like [AMHARIC],
    # but "|" is encoded as a regular token (mapped to space in vocab)
    batch["clean_transcription"] = [
        f"[{lang.upper()}]|{trans}"
        for lang, trans in zip(batch["language"], batch["clean_transcription"])
    ]

    return batch


def build_vocabulary(character_set: set[str],
                     add_language_tags: bool = False,
                     language_tags: List[str] = None) -> Dict[str, int]:
    """Build vocabulary from user-provided character set.
    
    Args:
        character_set: Set of characters to include in the vocabulary
        add_language_tags: Whether to add language tags to the vocabulary (only for multilingual models)
        language_tags: List of language tags to add to the vocabulary (only for multilingual models)
        
    Returns:
        Vocabulary dictionary
    """
    # create vocabulary dictionary from the user-provided character set
    vocab_dict = {v: k for k, v in enumerate(sorted(character_set))}

    # handle special tokens
    # add word delimiter token and remove space token
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # add unknown token and padding token
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # add language tag tokens to the vocabulary if specified
    # this can only be applied to multilingual models
    if add_language_tags:
        for tag in language_tags:
            tag_key = f"[{tag.upper()}]"
            vocab_dict[tag_key] = len(vocab_dict)
    
    return vocab_dict


def create_processor(
        config: ASRConfig, 
        vocab_path: str) -> ASRProcessor:
    """Create a processor from tokenizer and feature extractor.
    
    Args:
        vocab_path: Path to directory containing vocabulary file
        
    Returns:
        Wav2Vec2Processor for processing audio and text
    """
    # initialize tokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    
    # initialize feature extractor
    if config.pretrained_model == "facebook/w2v-bert-2.0":
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        # combine into processor
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer
        )

    else:
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        # combine into processor
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
    
    return processor


def prepare_datasets(train_dataset: Dataset, 
                     eval_dataset: Dataset, 
                     processor: Wav2Vec2Processor) -> Tuple[Dataset, Dataset]:
    """Prepare datasets for training by adding processed inputs.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        processor: Wav2Vec2Processor for processing audio and text
        
    Returns:
        Tuple of prepared (train_dataset, test_dataset)
    """
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=train_dataset.column_names
    )

    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset_batch(batch, processor),
        batched=True,
        batch_size=32, # has to be based on available memory
        remove_columns=eval_dataset.column_names
    )   
    
    return train_dataset, eval_dataset
