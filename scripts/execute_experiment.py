"""
Script for executing psycholinguistics experiments with language models.

This module provides utilities for generating text with detailed logprobs
and running experiments from batch files, saving results to Excel.
"""

import torch
import json
import math
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


def generate_with_logprobs(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    device: str,
    top_k: int = 10,
    output_path: str = 'generation_log.json',
    do_sample: bool = False
) -> Dict[str, Any]:
    """
    Generate text with detailed log probabilities and save to JSON.
    
    This function generates text from a language model and captures detailed
    probability information for each generated token, including the top-k
    most likely tokens at each generation step.
    
    Args:
        model: The loaded language model (e.g., from transformers).
        tokenizer: The tokenizer corresponding to the model.
        messages: List of message dicts with 'role' and 'content' keys.
                  Example: [{"role": "user", "content": "Hello!"}]
        device: Device to run the model on ('cuda', 'cpu', etc.).
        top_k: Number of top tokens to save at each generation step (default: 10).
        output_path: Path to save the JSON output (default: 'generation_log.json').
        do_sample: Whether to use sampling instead of greedy decoding (default: False).
    
    Returns:
        Dict containing the generation results with keys:
            - prompt_text: The formatted prompt text
            - input_ids: List of input token IDs
            - generated_ids: List of generated token IDs
            - generated_text: The decoded generated text
            - tokens: List of dicts with detailed info per token (step, token_id,
                     token, logprob, prob, topk)
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("model_name")
        >>> tokenizer = AutoTokenizer.from_pretrained("model_name")
        >>> messages = [{"role": "user", "content": "Explain ESAC briefly."}]
        >>> result = generate_with_logprobs(
        ...     model, tokenizer, messages, device="cuda",
        ...     max_new_tokens=50, output_path="output.json"
        ... )
        >>> print(f"Generated: {result['generated_text']}")
    """
    # Preparar texto / tokens
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)
    input_ids = inputs["input_ids"]  # shape (1, input_len)
    
    # Generar con logprobs
    generate_kwargs = {
        **inputs,
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": do_sample,
        "max_new_tokens": 32768,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    outputs = model.generate(**generate_kwargs)
    
    # outputs.sequences: tensor (batch, input_len + gen_len)
    generated_seq = outputs.sequences[0]  # tensor
    input_len = input_ids.size(1)
    gen_len = generated_seq.size(0) - input_len  # número de tokens generados
    
    result_tokens = []
    # outputs.scores: tuple de length gen_len, cada elemento shape (batch, vocab_size)
    for step, score in enumerate(outputs.scores):
        # score: logits en ese paso (antes del softmax)
        logits = score[0]  # batch 0 -> shape (vocab_size,)
        logprobs = torch.log_softmax(logits, dim=-1)
        
        token_id = int(generated_seq[input_len + step].item())
        token_text = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False
        )
        token_logprob = float(logprobs[token_id].item())
        token_prob = float(math.exp(token_logprob))
        
        # top-k del paso
        topk_list = _build_topk_list(logprobs, top_k, tokenizer)
        
        result_tokens.append({
            "step": step,
            "token_id": token_id,
            "token": token_text,
            "logprob": token_logprob,
            "prob": token_prob,
            "topk": topk_list
        })
    
    # Montar JSON final
    result = {
        "prompt_text": text,
        "input_ids": input_ids[0].tolist(),
        "generated_ids": generated_seq[input_len:].tolist(),
        "generated_text": tokenizer.decode(
            generated_seq[input_len:],
            skip_special_tokens=True
        ),
        "tokens": result_tokens
    }
    
    # Guardar a fichero JSON
    _save_json(result, output_path)
    print(f"Guardado en {output_path}")
    
    return result


def _save_json(data: Dict[str, Any], file_path: str) -> None:
    """Helper function to save data as JSON with UTF-8 encoding."""
    output_file = Path(file_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Helper function to read JSONL files."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _calculate_avg_logprob(tokens: List[Dict[str, Any]]) -> float:
    """Calculate average logprob from token list."""
    if not tokens:
        return 0.0
    return sum(t['logprob'] for t in tokens) / len(tokens)


def _build_topk_list(logprobs: torch.Tensor, k: int, tokenizer) -> List[Dict[str, Any]]:
    """Build top-k token list with probabilities."""
    topk = torch.topk(logprobs, k)
    return [
        {
            "id": int(tid),
            "token": tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False),
            "logprob": float(logprob),
            "prob": float(math.exp(float(logprob))),
        }
        for tid, logprob in zip(topk.indices.tolist(), topk.values.tolist())
    ]


def _check_incomplete_rows(df: pd.DataFrame) -> List[str]:
    """Check for incomplete rows and return list of words with missing data."""
    incomplete_words = []
    
    for idx, row in df.iterrows():
        word = row.get('word', '')
        response = row.get('response', '')
        logprob = row.get('logprob', None)
        
        # Check if word is missing or empty
        if pd.isna(word) or str(word).strip() == '':
            incomplete_words.append(f"Fila {idx+1}: palabra vacía")
            continue
        
        # Check if response is missing or empty
        if pd.isna(response) or str(response).strip() == '':
            incomplete_words.append(str(word))
            continue
        
        # Check if response starts with "ERROR:" (failed generation)
        if str(response).startswith('ERROR:'):
            incomplete_words.append(str(word))
            continue
        
        # Check if logprob is missing (but allow 0.0)
        if pd.isna(logprob):
            incomplete_words.append(str(word))
            continue
    
    return incomplete_words


def _get_incomplete_words_from_file(output_file: Path) -> Set[str]:
    """Get set of words that need to be reprocessed due to incomplete data."""
    if not output_file.exists():
        return set()
    
    try:
        df = pd.read_excel(output_file)
        incomplete_rows = _check_incomplete_rows(df)
        
        # Extract actual word names (filter out error messages like "Fila X: palabra vacía")
        incomplete_words = set()
        for item in incomplete_rows:
            if not item.startswith('Fila '):
                incomplete_words.add(item)
        
        return incomplete_words
        
    except Exception:
        return set()


def _validate_output_file(output_file: Path) -> bool:
    """Validate that the output file is properly formatted and readable."""
    if not output_file.exists():
        return True  # File doesn't exist, that's fine
    
    try:
        df = pd.read_excel(output_file)
        required_columns = ['word', 'response', 'logprob']
        
        # Check if all required columns exist
        if not all(col in df.columns for col in required_columns):
            print(f"Archivo {output_file.name} no tiene las columnas requeridas: {required_columns}")
            return False
        
        # Check if there are any duplicate words
        if df['word'].duplicated().any():
            duplicates = df[df['word'].duplicated()]['word'].tolist()
            print(f"Palabras duplicadas encontradas en {output_file.name}: {duplicates[:5]}...")
            return False
        
        # Check for incomplete rows (missing data)
        incomplete_rows = _check_incomplete_rows(df)
        if incomplete_rows:
            print(f"Se encontraron {len(incomplete_rows)} filas incompletas en {output_file.name}")
            print(f"   Primeras 5 palabras con datos incompletos: {incomplete_rows[:5]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validando archivo {output_file.name}: {e}")
        return False


def run_experiment(
    experiment_name: str,
    experiment_path: str,
    model,
    tokenizer,
    device: str,
    top_k: int = 10,
    do_sample: bool = False
) -> pd.DataFrame:
    """
    Execute a psycholinguistics experiment by processing prompts from a batch file.
    
    This function reads prompts from a JSONL batch file, generates responses with
    logprobs, and saves results to an Excel file. It automatically skips words
    that have already been processed (if output file exists).
    
    Args:
        experiment_name: Name of the experiment (e.g., 'familiarity_de').
                        Used to find the batch file: {experiment_name}.jsonl
        experiment_path: Path to the experiment folder (e.g., 'familiarity_german').
        model: The loaded language model.
        tokenizer: The tokenizer corresponding to the model.
        device: Device to run the model on ('cuda', 'cpu', etc.).
        top_k: Number of top tokens to save at each step (default: 10).
        do_sample: Whether to use sampling instead of greedy decoding (default: False).
    
    Returns:
        pd.DataFrame with columns: 'word', 'response', 'logprob'
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("model_name")
        >>> tokenizer = AutoTokenizer.from_pretrained("model_name")
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)
        >>> 
        >>> df = run_experiment(
        ...     experiment_name="familiarity_de",
        ...     experiment_path="familiarity_german",
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     device=device
        ... )
        >>> print(f"Processed {len(df)} words")
    """
    # Configurar rutas
    exp_path = Path(experiment_path)
    batch_file = exp_path / "batches" / f"{experiment_name}.jsonl"
    
    # Crear carpeta outputs si no existe
    outputs_dir = exp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = outputs_dir / f"{experiment_name}.xlsx"
    
    # Validar archivo de salida existente
    if not _validate_output_file(output_file):
        backup_file = outputs_dir / f"{experiment_name}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        if output_file.exists():
            print(f"Respaldando archivo problemático a: {backup_file.name}")
            output_file.rename(backup_file)
        processed_words = set()
        existing_data = []
    else:
        # Cargar palabras ya procesadas si existe el archivo de salida
        processed_words: Set[str] = set()
        existing_data = []
    
    if output_file.exists():
        print(f"Cargando resultados existentes desde: {output_file}")
        try:
            existing_df = pd.read_excel(output_file)
            if 'word' in existing_df.columns and len(existing_df) > 0:
                # Obtener todas las palabras del archivo
                all_words_in_file = set(existing_df['word'].tolist())
                
                # Obtener palabras incompletas que necesitan reprocesarse
                incomplete_words = _get_incomplete_words_from_file(output_file)
                
                # Solo considerar como procesadas las palabras completas
                processed_words = all_words_in_file - incomplete_words
                existing_data = existing_df.to_dict('records')
                
                print(f"Se encontraron {len(all_words_in_file)} palabras en el archivo")
                if incomplete_words:
                    print(f"{len(incomplete_words)} palabras incompletas serán reprocesadas")
                    print(f"   Ejemplos: {list(incomplete_words)[:5]}...")
                print(f"{len(processed_words)} palabras completamente procesadas")
            else:
                print("Archivo existe pero está vacío o no tiene la columna 'word'")
        except Exception as e:
            print(f"Error al cargar archivo existente: {e}")
            print("  Se procederá a crear un nuevo archivo")
            # Limpiar variables en caso de error
            processed_words = set()
            existing_data = []
    
    # Leer el archivo de batch
    if not batch_file.exists():
        raise FileNotFoundError(f"Archivo de batch no encontrado: {batch_file}")
    
    print(f"\nLeyendo prompts desde: {batch_file}")
    batch_data = _read_jsonl(batch_file)
    print(f"Total de prompts en batch: {len(batch_data)}")
    
    # Filtrar palabras ya procesadas
    pending_data = [item for item in batch_data if item['word'] not in processed_words]
    
    # Validación de coherencia
    if existing_data:
        expected_total = len(existing_data) + len(pending_data)
        actual_batch_total = len(batch_data)
        if expected_total != actual_batch_total:
            print(f"Advertencia: Inconsistencia detectada")
            print(f"   Archivo existente: {len(existing_data)} palabras")
            print(f"   Pendientes: {len(pending_data)} palabras")
            print(f"   Total esperado: {expected_total}, Batch actual: {actual_batch_total}")
            print(f"   Esto puede indicar que el batch ha cambiado desde la última ejecución")
    
    if not pending_data:
        print("\nTodas las palabras ya han sido procesadas!")
        return pd.DataFrame(existing_data)
    
    print(f"Palabras pendientes de procesar: {len(pending_data)}")
    
    # Procesar cada prompt pendiente
    results = existing_data.copy()
    total_batches = len(batch_data)  # Total de batches en el experimento
    processed_count = len(existing_data)  # Batches ya procesados
    
    print("\nIniciando procesamiento...\n\nProcesando palabras:")
    
    for idx, item in enumerate(pending_data, start=1):
        word = item['word']
        prompt = item['prompt']
        
        # Mostrar progreso en la misma línea
        current_batch = processed_count + idx  # Batch actual en proceso
        progress_percent = (current_batch / total_batches) * 100
        progress_bar = "█" * int(progress_percent // 5) + "░" * (20 - int(progress_percent // 5))
        print(f"\r[{progress_bar}] {progress_percent:.1f}% ({current_batch}/{total_batches}) Procesando: {word}", end="", flush=True)
        
        # Preparar mensajes para el modelo
        messages = [{"role": "user", "content": prompt}]
        
        # Generar respuesta con logprobs
        try:
            generation_result = _generate_single_response(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                device=device,
                top_k=top_k,
                do_sample=do_sample
            )
            
            response = generation_result['generated_text']
            avg_logprob = _calculate_avg_logprob(generation_result['tokens'])
            
            results.append({
                'word': word,
                'response': response,
                'logprob': avg_logprob
            })
            
        except Exception as e:
            results.append({
                'word': word,
                'response': f"ERROR: {str(e)}",
                'logprob': None
            })
        
        # Guardar después de cada palabra (para no perder progreso)
        try:
            df_results = pd.DataFrame(results)
            
            # Guardar con backup temporal
            temp_file = output_file.with_suffix('.tmp')
            df_results.to_excel(temp_file, index=False)
            
            # Si el guardado temporal fue exitoso, reemplazar el archivo original
            if temp_file.exists():
                if output_file.exists():
                    output_file.unlink()  # Eliminar archivo original
                temp_file.rename(output_file)  # Renombrar temp a original
            
            # Actualizar processed_words para mantener sincronización
            processed_words.add(word)
            
        except Exception as save_error:
            # Continuar con la siguiente palabra sin imprimir error para mantener barra de progreso
            pass
    
    # Completar la barra de progreso
    print(f"\r[{'█' * 20}] 100.0% ({total_batches}/{total_batches}) ¡Completado!                                          ")
    
    print(f"\nProcesamiento completado!")
    
    # Validación final del archivo resultante
    final_df = pd.DataFrame(results)
    incomplete_final = _check_incomplete_rows(final_df)
    
    if incomplete_final:
        print(f"\nADVERTENCIA: El experimento se completó pero {len(incomplete_final)} palabras tienen datos incompletos:")
        print(f"   {incomplete_final[:10]}...")  # Mostrar las primeras 10
        print(f"   Estas palabras serán reprocesadas en la próxima ejecución.")
    else:
        print(f"\nÉXITO: Todas las {len(final_df)} palabras han sido procesadas correctamente.")
        print("   No se encontraron datos incompletos.")
    
    print(f"\nResumen final:")
    print(f"   Total de palabras en archivo: {len(final_df)}")
    if incomplete_final:
        complete_words = len(final_df) - len(incomplete_final)
        print(f"   Palabras completamente procesadas: {complete_words}")
        print(f"   Palabras con datos incompletos: {len(incomplete_final)}")
    
    return final_df


def _generate_single_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    device: str,
    top_k: int = 10,
    do_sample: bool = False
) -> Dict[str, Any]:
    """
    Generate a single response with logprobs (internal helper function).
    
    Similar to generate_with_logprobs but without file saving.
    Returns the generation result dict.
    """
    # Suprimir warnings de transformers (pad_token_id)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")
        
        # Preparar texto / tokens
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)
        input_ids = inputs["input_ids"]
        
        # Generar con logprobs - permitir al modelo generar libremente
        outputs = model.generate(
            **inputs,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=do_sample,
            max_new_tokens=32768,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_seq = outputs.sequences[0]
        input_len = input_ids.size(1)
        gen_len = generated_seq.size(0) - input_len
        
        result_tokens = []
        for step, score in enumerate(outputs.scores):
            logits = score[0]
            logprobs = torch.log_softmax(logits, dim=-1)
            
            token_id = int(generated_seq[input_len + step].item())
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            token_logprob = float(logprobs[token_id].item())
            token_prob = float(math.exp(token_logprob))
            
            # top-k del paso
            topk_list = _build_topk_list(logprobs, top_k, tokenizer)
            
            result_tokens.append({
                "step": step,
                "token_id": token_id,
                "token": token_text,
                "logprob": token_logprob,
                "prob": token_prob,
                "topk": topk_list
            })
        
        result = {
            "prompt_text": text,
            "input_ids": input_ids[0].tolist(),
            "generated_ids": generated_seq[input_len:].tolist(),
            "generated_text": tokenizer.decode(
                generated_seq[input_len:],
                skip_special_tokens=True
            ),
            "tokens": result_tokens
        }
        
        return result


# Example usage (commented out):
"""
Example 1: Run experiment from main.ipynb or Python script
-----------------------------------------------------------
from scripts.execute_experiment import run_experiment
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "your-model-name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Run experiment
df_results = run_experiment(
    experiment_name="familiarity_de",
    experiment_path="familiarity_german",
    model=model,
    tokenizer=tokenizer,
    device=device
)

print(df_results.head())


Example 2: Generate with full logprobs and save to JSON
--------------------------------------------------------
from scripts.execute_experiment import generate_with_logprobs
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-model-name"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = "Give me a brief explanation of ESAC in simple terms."
messages = [{"role": "user", "content": prompt}]

result = generate_with_logprobs(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    device=device,
    max_new_tokens=128,
    top_k=10,
    output_path='generation_log.json'
)

print(f"Generated text: {result['generated_text']}")
print(f"Number of tokens generated: {len(result['tokens'])}")
"""
