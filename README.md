# python-lesson-agent

## Installation

1) Download and install ollama from [ollama.com](https://ollama.com).

2) Pull model for ollama from hf:
```bash
ollama run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
```
3) pip install -r requirements.txt

4) unrar archive from https://drive.google.com/file/d/1dNCCDHOPbBxts2CrRmVwlfQ0PMhER17N/view?usp=sharing and put folder qdrant_storage to repo root

## Use

firstly,
```
./start_services.sh
```
and then inference:
```
python run.py "description_of_your_lesson"
```

## Your data

it was used script src/data_ingestion/ingest.py to scrape the data from docs and books, if you want to use your own data, then look at ingest.py code
