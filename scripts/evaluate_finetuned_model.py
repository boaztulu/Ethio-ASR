#!/usr/bin/env python3

import torch
from transformers import AutoProcessor, AutoModelForCTC
from datasets import load_from_disk, Audio, Dataset, load_dataset
import jiwer
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import evaluate
import re
from typing import Dict, List, Any, Tuple
import logging
import sys
sys.path.insert(0, '/nethome/badr/projects/arabic-speech-dialectness/kinyarwanda-ASR')

from src.data.preprocessing import _clean_text_with_set

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

# def process_text(text: str) -> str:
#     """
#     Text normalization:
#     1. Remove punctuation, except Kinayarwanda's apostrophe (')
#     2. Remove digits
#     3. Trim leading/trailing whitespace
#     """

#     # keep only allowed characters
#     #allowed = "abcdefghijklmnopqrstuvwxyz \'"
#     allowed = " abcdefghijklmnopqrstuvwxyz-'0123456789ሀሁሂሃሄህሆለሉሊላሌልሎሐሑሒሓሔሕሖመሙሚማሜምሞሠሡሢሣሤሥሦረሩሪራሬርሮሰሱሲሳሴስሶሸሹሺሻሼሽሾቀቁቂቃቄቅቆበቡቢባቤብቦቨቩቪቫቬቭቮተቱቲታቴትቶቸቹቺቻቼችቾኀኁኂኃኄኅኆነኑኒናኔንኖኘኙኚኛኜኝኞአኡኢኣኤእኦኧከኩኪካኬክኮኸኹኺኻኼኽኾወዉዊዋዌውዎዐዑዒዓዔዕዖዘዙዚዛዜዝዞየዩዪያዬይዮደዱዲዳዴድዶጀጁጂጃጄጅጆገጉጊጋጌግጎጠጡጢጣጤጥጦጨጩጪጫጬጭጮጰጱጲጳጴጵጶጸጹጺጻጼጽጾፀፁፂፃፄፅፆፈፉፊፋፌፍፎፐፑፒፓፔፕፖቐቑቒቓቔቕቖቘ቙ቚቛቜቝኰ኱ኲኳኴኵዀ዁ዂዃዄዅ዆ጐ጑ጒጓጔጕጓዸዹዺዻዼዽዾ"
#     text = ''.join(c for c in text if c.lower() in allowed)
    
#     # normalize whitespace
#     text = re.sub(r'\s+', ' ', text).strip()

#     return text.lower().strip()

def process_text(text: str) -> str:
    allowed = " abcdefghijklmnopqrstuvwxyz-'0123456789ሀሁሂሃሄህሆለሉሊላሌልሎሐሑሒሓሔሕሖመሙሚማሜምሞሠሡሢሣሤሥሦረሩሪራሬርሮሰሱሲሳሴስሶሸሹሺሻሼሽሾቀቁቂቃቄቅቆበቡቢባቤብቦቨቩቪቫቬቭቮተቱቲታቴትቶቸቹቺቻቼችቾኀኁኂኃኄኅኆነኑኒናኔንኖኘኙኚኛኜኝኞአኡኢኣኤእኦኧከኩኪካኬክኮኸኹኺኻኼኽኾወዉዊዋዌውዎዐዑዒዓዔዕዖዘዙዚዛዜዝዞየዩዪያዬይዮደዱዲዳዴድዶጀጁጂጃጄጅጆገጉጊጋጌግጎጠጡጢጣጤጥጦጨጩጪጫጬጭጮጰጱጲጳጴጵጶጸጹጺጻጼጽጾፀፁፂፃፄፅፆፈፉፊፋፌፍፎፐፑፒፓፔፕፖቐቑቒቓቔቕቖቘ቙ቚቛቜቝኰ኱ኲኳኴኵዀ዁ዂዃዄዅ዆ጐ጑ጒጓጔጕጓዸዹዺዻዼዽዾ"
    return _clean_text_with_set(text, set(allowed), apply_accent_replacements=True)

# type alias for cleaner signature
ModelComponents = Tuple[AutoModelForCTC, AutoProcessor, torch.device]

def load_model(model_path: str) -> ModelComponents:
    
    """Load the trained ASR model and processor from local path"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_audio(audio_array: Audio, 
                     model: AutoModelForCTC, 
                     processor: AutoProcessor, 
                     device: torch.device) -> str:
    
    """Transcribe a single audio array"""
    # make sure audio is at 16kHz sampling rate
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=False # this should ignore special tokens like [PAD]
    )

    return transcription[0].strip()


def evaluate_split(dataset: Dataset, 
                   model: AutoModelForCTC, 
                   processor: AutoProcessor, 
                   device: torch.device,
                   text_column: str='transcription', 
                   split_name='validation') -> Dict[str, Any]:

    """Evaluate a dataset split and calculate WER/CER"""
    eval_dataset = dataset

    # if not eval_dataset:
    #     print(f"No data found for {split_name} split.")
    #     return {}

    print(f"Processing {len(eval_dataset)} samples...")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []

    # resample dataset to 16kHz if needed
    if eval_dataset.features['audio'].sampling_rate != 16000:
        print(f"Resampling speech to 16kHz...")

        eval_dataset = eval_dataset.cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
    
    for i, sample in enumerate(tqdm(eval_dataset, desc=f"Transcribing speech.")):
        # Get audio array (should be at 16kHz)
        audio_array = sample['audio']['array']
        
        # Transcribe
        pred_text = transcribe_audio(audio_array, model, processor, device)

        # remove [PAD] tokens from prediction
        # this is a hack for now, should be investigated and fixed properly
        #pred_text = pred_text.replace("[PAD]", "").strip()

        #pred_text = transcribe_audio(audio_array, model, processor, device)


        # Use cleaned_text as reference
        ref_text_1 = process_text(sample[text_column])

        # encode then decode reference (same as training)
        ref_encoded = processor.tokenizer(process_text(sample[text_column]))
        ref_text_2 = processor.tokenizer.decode(ref_encoded.input_ids)
        ref_text_3 = processor.tokenizer.decode(ref_encoded.input_ids, group_tokens=False)


        print(f"Prediction: {pred_text}")
        print(f"Reference 1: {ref_text_1}")
        print(f"Reference 2: {ref_text_2}")
        print(f"Reference 3: {ref_text_3}")

        predictions.append(pred_text)
        references.append(ref_text_1)

    # print total number of predictions and references
    print(f"Total predictions: {len(predictions)}, Total references: {len(references)}")

    for i in range(len(references)):
        if not references[i]:
            print(f"Empty Reference at index {i}: {references[i]}")
            print(f"Prediction: {predictions[i]}")


    # Filter out empty predictions and references
    filtered_predictions = []
    filtered_references = []

    for pred, ref in zip(predictions, references):
        if ref:
            filtered_predictions.append(pred)
            filtered_references.append(ref)
            #print(f"Reference: {ref}")
            #print(f"Prediction: {pred}")

    # print total number of filtered predictions and references
    print(f"Filtered predictions: {len(filtered_predictions)}, Filtered references: {len(filtered_references)}")


    wer = wer_metric.compute(predictions=filtered_predictions, references=filtered_references)
    cer = cer_metric.compute(predictions=filtered_predictions, references=filtered_references)

    error_rate = (0.4 * wer) + (0.6 * cer)
    score = (1 - error_rate) * 100

    print(f"{split_name} Results:")
    print(f"  WER: {wer:.4f} ({wer*100:.4f}%)")
    print(f"  CER: {cer:.4f} ({cer*100:.4f}%)")
    print(f"  Score: {score:.4f}%")

    
    return {
        'split': split_name,
        'wer': wer,
        'cer': cer,
        'score': score,
        'samples': len(eval_dataset),
        'predictions': predictions,
        'references': references
    }


def main():
    # Setup logging
    setup_logging()
    logging.info("Starting evaluation script...")

    # Configuration
    #model_path = "./inprogress/baseline/facebook/w2v-bert-2.0-20062025-203510/checkpoint-2400"  
    #model_path = "./inprogress/current_best/checkpoint-13200/"
    model_path = "./inprogress/ethiopian-ASR/facebook/w2v-bert-2.0-24112025-163735"

    print("Loading trained model from {model_path}...")
    model, processor, device = load_model(model_path)

    print(f"Model loaded on device: {device}")

    dataset_path = "badrex/ethiopian-speech-flat"

    text_column = 'transcription' #'transcription' 
    split = 'validation' 

    logging.info(f"Loading dataset {dataset_path}/{split}...")
    dataset = load_dataset(dataset_path, "all")
    dataset = dataset[split]

    results = {}
    
    results[split] = evaluate_split(
        dataset, model, processor, device, text_column, split
    )

    
    # # If no validation/test splits, evaluate a subset of train split
    # if not results:
    #     print("No validation/test splits found. Evaluating subset of train split...")
    #     train_subset = dataset['train'].select(range(min(1000, len(dataset['train']))))
    #     results['train_subset'] = evaluate_split(
    #         train_subset, model, processor, device, 'train_subset'
    #     )
    
    # make a summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    summary_data = []
    for split_name, result in results.items():
        summary_data.append({
            'Split': split_name,
            'Samples': result['samples'],
            'WER (%)': f"{result['wer']*100:.4f}%",
            'CER (%)': f"{result['cer']*100:.4f}%",
            'Score (%)': f"{result['score']:.4f}%",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # # Save detailed results
    # for split_name, result in results.items():
    #     results_df = pd.DataFrame({
    #         'reference': result['references'],
    #         'prediction': result['predictions']
    #     })
    #     results_df.to_csv(f"{split_name}_results.tsv", sep='\t', index=False)
    #     print(f"\nDetailed results saved to {split_name}_results.csv")

if __name__ == "__main__":
    main()


"""
with skipe_special_tokens=True, the output is:
==================================================
EVALUATION SUMMARY
==================================================
     Split  Samples  WER (%) CER (%) Score (%)
validation     4632 18.3810% 3.9781%  90.2607%

without skipe_special_tokens=True, the output is:
==================================================
EVALUATION SUMMARY
==================================================
     Split  Samples  WER (%) CER (%) Score (%)
validation     4632 18.3810% 3.9781%  90.2607%

2800
==================================================
EVALUATION SUMMARY
==================================================
     Split  Samples  WER (%) CER (%) Score (%)
validation     4632 15.6370% 3.3036%  91.7631%

6800
==================================================
EVALUATION SUMMARY
==================================================
     Split  Samples  WER (%) CER (%) Score (%)
validation     4632 11.0892% 2.4881%  94.0715%

13200
==================================================
EVALUATION SUMMARY
==================================================
     Split  Samples WER (%) CER (%) Score (%)
validation     4632 9.3677% 2.1123%  94.9855%


common voice dataset:
test Results:
  WER: 0.3960 (39.6027%)
  CER: 0.1252 (12.5244%)
  Score: 76.6443%

==================================================
EVALUATION SUMMARY
==================================================
Split  Samples  WER (%)  CER (%) Score (%)
 test    16213 39.6027% 12.5244%  76.6443%


"""