# transcribe wolaytta validation set and save results
import os

from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import evaluate
import re

from tqdm import tqdm

# get parent directory and add to sys.path
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from scripts.evaluate_finetuned_model_mutlilingual import process_text

lang2code = {
    "afaan_oromo": "orm_Latn",
    "wolaytta": "wol_Latn",
    "amharic": "amh_Ethi",
    "tigrinya": "tir_Ethi",
    "sidama": "sid_Latn",
}

transcript_dir = "transcripts/omni_asr"
ds = load_dataset("badrex/ethiopian-speech-flat", "all")
ds = ds["validation"]

lang = "sidama"


ds = ds.filter(lambda x: x["language"] == lang, num_proc=8)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# 
#models = ["omniASR_CTC_300M_v2", "omniASR_CTC_1B_v2", "omniASR_CTC_3B_v2"] 
models = ["omniASR_LLM_300M_v2", "omniASR_LLM_1B_v2", "omniASR_LLM_3B_v2"] 

# load model
for model in models:
    pipeline = ASRInferencePipeline(model_card=model)

    # transcribe all samples
    results = []
    references = []
    predictions = []

    for i, sample in tqdm(enumerate(ds), total=len(ds)):
        audio = {"waveform": sample["audio"]["array"], "sample_rate": sample["audio"]["sampling_rate"]}
        pred = process_text(pipeline.transcribe([audio], lang=[lang2code[lang]], batch_size=1)[0])
        ref = process_text(sample["transcription"])
        results.append({"id": i, "reference": ref, "prediction": pred})
        
        print(f"Sample {i}:")
        print(f"Prediciton: {pred}")
        print(f" Reference: {ref.lower()}")
        print("-"*75)

        predictions.append(pred)
        references.append(ref.lower())

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print(f"WER: {wer*100:.4f}%")
    print(f"CER: {cer*100:.4f}%")
    print(f"Score: {(1 - (0.5 * wer + 0.5 * cer))*100:.4f}%")


    # write to filee
    with open(f"{transcript_dir}/{lang}_{model}_validation_predictions.tsv", "w") as f:
        f.write("id\treference\tprediction\n")
        for r in results:
            f.write(f"{r['id']}\t{r['reference']}\t{r['prediction']}\n")

    print(f"Saved {len(results)} samples to {transcript_dir}/{lang}_{model}_validation_predictions.tsv")