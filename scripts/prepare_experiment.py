"""

It reads the experiment config, loads the dataset column, injects each word into the prompt
template (placeholder {WORD}) and writes a JSONL file at <experiment_path>/batches/<experiment_basename>.jsonl

"""

import json
import os
import jsonlines
import re

from scripts.utils import load_config, read_txt, read_column_as_list, read_yaml


def build_prompt_list_from_template(word_list: list, prompt_template: str, experiment_name: str = None) -> list:
    prompts = []
    for idx, w in enumerate(word_list, start=1):
        prompt_text = re.sub(r"\{word\}", str(w), prompt_template, flags=re.IGNORECASE)
        prompt_text = prompt_text.strip()
        if experiment_name:
            custom_id = f"{experiment_name}_task_{idx}"
        else:
            custom_id = f"task_{idx}"
        prompts.append({"custom_id": custom_id, "prompt": prompt_text, "word": w})
    return prompts


def create_batches(tasks: list, run_prefix: str, chunk_size: int = 50000, output_name: str = None):
    os.makedirs(run_prefix, exist_ok=True)
    batches_dir = os.path.join(run_prefix, "batches")
    os.makedirs(batches_dir, exist_ok=True)
    filename_base = output_name if output_name else os.path.basename(run_prefix)
    batch_file = os.path.join(batches_dir, f"{filename_base}.jsonl")
    
    try:
        with jsonlines.open(batch_file, "w") as writer:
            for item in tasks:
                writer.write(item)
    except Exception:
        with open(batch_file, "w", encoding="utf-8") as f:
            for item in tasks:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return batch_file


def prepare_batches_for_experiment(experiment_name: str, experiment_path: str) -> str:
    try:
        config_args = load_config(config_type="experiments", name=experiment_name)
    except Exception:
        config_file = os.path.join(experiment_path, "config.yaml")
        if os.path.exists(config_file):
            conf = read_yaml(config_file)
            config_args = conf.get("experiments", {}).get(experiment_name, {})
        else:
            conf = read_yaml(os.path.join("familiarity_german", "config.yaml"))
            config_args = conf.get("experiments", {}).get(experiment_name, {})

    if not config_args:
        raise ValueError(f"Experiment {experiment_name} not found in config")

    dataset_path = os.path.join(experiment_path, "data", config_args["dataset_path"]) if not os.path.isabs(config_args["dataset_path"]) else config_args["dataset_path"]
    dataset_column = config_args.get("dataset_column", "Word")
    prompt_path = os.path.join(experiment_path, "prompts", config_args["prompt_path"]) if not os.path.isabs(config_args["prompt_path"]) else config_args["prompt_path"]

    print(f"Preparando experimento '{experiment_name}'...")
    
    # Paso 1: Leer palabras
    print(f"\r[░░░░░░░░░░░░░░░░░░░░] 0.0% Leyendo dataset...", end="", flush=True)
    words = read_column_as_list(dataset_path, dataset_column)
    total_words = len(words)
    
    # Paso 2: Leer template
    print(f"\r[██░░░░░░░░░░░░░░░░░░] 10.0% Leyendo template de prompt...", end="", flush=True)
    prompt_template = read_txt(prompt_path)
    
    # Paso 3: Generar prompts
    print(f"\r[████░░░░░░░░░░░░░░░░] 20.0% Generando prompts para {total_words} palabras...", end="", flush=True)
    prompts = build_prompt_list_from_template(words, prompt_template, experiment_name=experiment_name)
    
    # Paso 4: Guardar archivo
    print(f"\r[████████████████████] 100.0% Guardando archivo JSONL...", end="", flush=True)
    batch_path = create_batches(prompts, run_prefix=experiment_path, output_name=experiment_name)
    
    # Completar
    print(f"\r[████████████████████] 100.0% ¡Completado!                                          ")
    print(f"{total_words} prompts preparados y guardados en: {batch_path}")
    
    return batch_path


