
# Ethio ASR

To run the code, follow the instructions below 

### 1. Clone the repo
```shell
clone https://github.com/badrex/Ethio-ASR.git
```

### 2. Set up environment variables in .env

Then create an empty  `.env` file in the project root:
```env
# Weights & Biases API key
WANDB_API_KEY="wandb_api_key_xxx"

# Hugging Face API key
HF_API_KEY="hf_api_key_xxx"

# Set cache
NUMBA_CACHE_DIR='/tmp/numba_cache'
LIBROSA_CACHE_DIR="/tmp/librosa_cache"

# Hugging Face cache directory
HF_HOME='/project_dir/huggingface_cache'

# MPLCONFIGDIR
MPLCONFIGDIR='/tmp/matplotlib_cache'
```


### 3. Set the YAML config file to run the experiment 
This config is under config_files. This config below is for an experiment on a small scale dataset.

```yaml
# Project settings
# this is the WANDB project name
project: "Ethio-ASR"

# this is the output directory for  saving the model and processor 
output_dir: "inprogress/Ethio-ASR"

# set random seed for reproducibility
seed: 42

# Model settings
pretrained_model:   "facebook/w2v-bert-2.0" # or "acebook/mms-300m  
freeze_feature_encoder: true
add_final_layer_adapter: true # should be false for facebook/mms-300m

# Training settings
batch_size: 8
gradient_accumulation_steps: 4
num_epochs: 25
max_steps: 18400
learning_rate: 0.00003 # or 0.0005 for "facebook/mms-300m"  
warmup_ratio: 0.1
fp16: true
gradient_checkpointing: true
save_steps: 800
eval_steps: 800
logging_steps: 5
save_total_limit: 2

# Data settings
# if use_custom_dataset is true, then dataset_path is the path to the custom dataset on disk
# if use_custom_dataset is false, then dataset_path is the dataset repo name on the HF hub
use_custom_dataset: false
# if from HF hub, use the repo name for example "badrex/waxalNLP-ethiopic-final" 
dataset_path: "badrex/waxalNLP-ethiopic-final"   
train_split: "train"
eval_split: "validation"
language: "all"

# Data sampling settings (for debugging purposes)
sample: true
sample_size: 197634 # this is the size of the dataset in the HF hub


## Text preprocessing settings
# Character set allowed in transcriptions - customize based on language and script
add_language_tokens: true
apply_accent_replacements: true
character_set: " !#$%&'*+,-.0123456789=?@abcdefghijklmnopqrstuvwxyzሀሁሂሃሄህሆሇለሉሊላሌልሎሏሐሑሒሓሔሕሖሗመሙሚማሜምሞሟሠሡሢሣሤሥሦሧረሩሪራሬርሮሯሰሱሲሳሴስሶሷሸሹሺሻሼሽሾሿቀቁቂቃቄቅቆቇቈቊቋቌቍቐቑቒቓቔቕቖቘቚቛቜቝበቡቢባቤብቦቧቨቩቪቫቬቭቮቯተቱቲታቴትቶቷቸቹቺቻቼችቾቿኀኁኂኃኄኅኆኇኈኊኋኌኍነኑኒናኔንኖኗኘኙኚኛኜኝኞኟአኡኢኣኤእኦኧከኩኪካኬክኮኯኰኲኳኴኵኸኹኺኻኼኽኾዀዂዃዄዅወዉዊዋዌውዎዏዐዑዒዓዔዕዖዘዙዚዛዜዝዞዟዠዡዢዣዤዥዦዧየዩዪያዬይዮዯደዱዲዳዴድዶዷዸዹዺዻዼዽዾዿጀጁጂጃጄጅጆጇገጉጊጋጌግጎጏጐጒጓጔጕጘጙጚጛጜጝጞጟጠጡጢጣጤጥጦጧጨጩጪጫጬጭጮጯጰጱጲጳጴጵጶጷጸጹጺጻጼጽጾጿፀፁፂፃፄፅፆፇፈፉፊፋፌፍፎፏፐፑፒፓፔፕፖፗፘፙፚ፠፡።፣፤፥፦፧፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼€"
```



### 4. Set bash script
This is the main script for running the experiment. The most important is to setup the correct path to the Hugging Face directory. 

The script below run a debugging experiment on a small scale dataset.

```shell
##!/usr/bin/env bash

# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# cahce dirs 
export HF_HOME="/project_dir/huggingface_cache"
NUMBA_CACHE_DIR='/tmp/numba_cache'
LIBROSA_CACHE_DIR="/tmp/librosa_cache"

# show current working directory
echo "Current working directory: $(pwd)"

# run training script
python3 Ethio-ASR/scripts/train_model.py --config Ethio-ASR/config_files/ASR_train_config_multi_debug.yaml
```


### 5. Submit the run.sub file


```shell
universe              = docker
docker_image          = badrnlp/hf-gpu-asr:0.2

transfer_executable = False

initialdir            = /project_dir/
executable            = /project_dir/Ethio-ASR/bash_scripts/run_train_ethio_asr.sh

output                = logs/run.sh.$(ClusterId).$(Month)_$(Day)_$(SUBMIT_TIME).out
error                 = logs/run.sh.$(ClusterId).$(Month)_$(Day)_$(SUBMIT_TIME).err
log                   = logs/run.sh.$(ClusterId).$(Month)_$(Day)_$(SUBMIT_TIME).log

request_CPUs          = 12
request_memory        = 50G
request_GPUs          = 1

# Limit to the failing node for testing
requirements          = (TARGET.UidDomain == "lsv.uni-saarland.de") && (machine == "cl18lx.lsv.uni-saarland.de")
queue 1
```