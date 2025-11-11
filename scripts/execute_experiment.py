"""

This module provides utilities for generating text responses and saving the results to Excel.

"""

import torch
import json
import math
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Set


def _read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Helper function to read JSONL files."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _calculate_avg_logprob(tokens: List[Dict[str, Any]]) -> float:
    """Calculate logprob score using pre-inference digit probability method."""
    if not tokens:
        return 0.0
    return _calculate_digit_logprob(tokens[0]['topk']) if tokens else 0.0


def _calculate_digit_logprob(topk_list: List[Dict[str, Any]]) -> float:
    """
    Calculate weighted logprob score from digits 1-7.
    
    score = Σ(digit × probability) for digits ∈ {1,2,3,4,5,6,7} ∩ top_10
    
    Args:
        topk_list: Top-k tokens
        
    Returns:
        Weighted sum of digit values by their probabilities
        
    Example:
        Top 10: "1"(0.9), "2"(0.05), "3"(0.05)
        Result: 1×0.9 + 2×0.05 + 3×0.05 = 1.15
    """
    digit_probs = {}
    
    # Find digits 1-7 in top-k tokens
    for token_info in topk_list:
        token = token_info['token'].strip()
        prob = token_info['prob']
        
        if token in ['1', '2', '3', '4', '5', '6', '7']:
            digit = int(token)
            if digit not in digit_probs:  # No duplicates
                digit_probs[digit] = prob
    
    # Calculate weighted sum
    return sum(digit * prob for digit, prob in digit_probs.items())


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
            incomplete_words.append(idx)
            continue
        
        # Check if response is missing or empty
        if pd.isna(response) or str(response).strip() == '':
            incomplete_words.append(idx)
            continue
        
        # Check if response starts with "ERROR:" (failed generation)
        if str(response).startswith('ERROR:'):
            incomplete_words.append(idx)
            continue
        
        # Check if logprob is missing (but allow 0.0)
        if pd.isna(logprob):
            incomplete_words.append(idx)
            continue
    
    return incomplete_words


def _get_incomplete_words_from_file(output_file: Path, batch_file: Path) -> Set[str]:
    """Get set of words that need to be reprocessed due to incomplete data."""
    if not output_file.exists():
        return set()
    if not batch_file.exists():
        return set()
    
    try:
        df = pd.read_excel(output_file)
        incomplete_rows = _check_incomplete_rows(df)

        batch_data = _read_jsonl(batch_file)
        
        # Extract actual word names from batches
        incomplete_words = set()
        for idx in incomplete_rows:
            if idx < len(batch_data):  # Validate index is within range
                word = batch_data[idx].get('word', '')
                if word and str(word).strip():  # Only add non-empty words
                    incomplete_words.add(str(word))
        
        return incomplete_words
        
    except Exception:
        return set()


def _validate_output_file(output_file: Path) -> bool:
    """Validate that the output file is properly formatted and readable."""
    if not output_file.exists():
        return True
    
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
            print(f"Palabras duplicadas encontradas en {output_file.name}: {duplicates}")
            return False
        
        # Check for incomplete rows (missing data)
        incomplete_rows = _check_incomplete_rows(df)
        if incomplete_rows:
            print(f"Se encontraron {len(incomplete_rows)} filas incompletas en {output_file.name}")
            print(f"   Palabras con datos incompletos: {incomplete_rows}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validando archivo {output_file.name}: {e}")
        return False


def run_experiment( experiment_name: str, experiment_path: str, model, tokenizer, device: str, top_k: int = 10, do_sample: bool = False) -> pd.DataFrame:
    """
    
    Processes prompts from JSONL batch file, generates responses, and calculates
    logprob scores based on digit probabilities (1-7) in the pre-inference
    distribution. Results are saved to Excel.
    
    Args:
        experiment_name: Experiment identifier (finds {experiment_name}.jsonl)
        experiment_path: Path to experiment folder
        model: Loaded language model
        tokenizer: Model tokenizer
        device: Device for model execution ('cuda' / 'cpu')
        top_k: Number of top tokens to analyze (default: 10)
        do_sample: Use sampling vs greedy decoding (default: False)
    
    Returns:
        Excel with columns: 'word', 'response', 'logprob'
    """
    # Setup paths
    exp_path = Path(experiment_path)
    batch_file = exp_path / "batches" / f"{experiment_name}.jsonl"
    
    # Create outputs directory
    outputs_dir = exp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = outputs_dir / f"{experiment_name}.xlsx"
    
    # Validate existing output file
    if not _validate_output_file(output_file):
        backup_file = outputs_dir / f"{experiment_name}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        if output_file.exists():
            print(f"Respaldando archivo problemático a: {backup_file.name}")
            output_file.rename(backup_file)
        processed_words = set()
        existing_data = []
    else:
        # Load processed words if output file exists
        processed_words: Set[str] = set()
        existing_data = []
    
    if output_file.exists():
        print(f"Cargando resultados existentes desde: {output_file}")
        try:
            existing_df = pd.read_excel(output_file)
            if 'word' in existing_df.columns and len(existing_df) > 0:
                # Get all words and identify incomplete ones
                all_words_in_file = set(existing_df['word'].tolist())
                incomplete_words = _get_incomplete_words_from_file(output_file, batch_file)
                
                # Only consider complete words as processed
                processed_words = all_words_in_file - incomplete_words
                existing_data = existing_df.to_dict('records')
                
                print(f"Se encontraron {len(all_words_in_file)} palabras en el archivo")
                if incomplete_words:
                    print(f"{len(incomplete_words)} palabras incompletas serán reprocesadas")
                print(f"{len(processed_words)} palabras completamente procesadas")
            else:
                print("Archivo existe pero está vacío o no tiene la columna 'word'")
        except Exception as e:
            print(f"Error al cargar archivo existente: {e}")
            print("  Se procederá a crear un nuevo archivo")
            processed_words = set()
            existing_data = []
    
    # Read batch file
    if not batch_file.exists():
        raise FileNotFoundError(f"Archivo de batch no encontrado: {batch_file}")
    
    print(f"\nLeyendo prompts desde: {batch_file}")
    batch_data = _read_jsonl(batch_file)
    print(f"Total de prompts en batch: {len(batch_data)}")
    
    # Filter already processed words
    pending_data = [item for item in batch_data if item['word'] not in processed_words]
    
    # Consistency validation
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
    
    # Process each pending prompt
    results = existing_data.copy()
    total_batches = len(batch_data)
    processed_count = len(existing_data)
    
    print("\nIniciando procesamiento...\n\nProcesando palabras:")
    
    for idx, item in enumerate(pending_data, start=1):
        word = item['word']
        prompt = item['prompt']
        
        # Show progress
        current_batch = processed_count + idx
        progress_percent = (current_batch / total_batches) * 100
        progress_bar = "█" * int(progress_percent // 5) + "░" * (20 - int(progress_percent // 5))
        print(f"\r[{progress_bar}] {progress_percent:.1f}% ({current_batch}/{total_batches}) Procesando: {word}", end="", flush=True)
        
        # Prepare messages for model
        messages = [{"role": "user", "content": prompt}]
        
        # Generate response
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
        
        # Save after each result to preserve progress
        try:
            df_results = pd.DataFrame(results)
            
            # Save with temporary backup
            temp_file = output_file.with_suffix('.tmp')
            df_results.to_excel(temp_file, index=False)
            
            # Replace original file if temp save was successful
            if temp_file.exists():
                if output_file.exists():
                    output_file.unlink()
                temp_file.rename(output_file)
            
            processed_words.add(word)
            
        except Exception as save_error:
            # Continue without printing error to maintain progress bar
            pass
    
    # Complete progress bar
    print(f"\r[{'█' * 20}] 100.0% ({total_batches}/{total_batches}) ¡Completado!")
    
    print(f"\nProcesamiento completado!")
    
    # Final validation
    final_df = pd.DataFrame(results)
    incomplete_final = _check_incomplete_rows(final_df)
    
    if incomplete_final:
        print(f"\nADVERTENCIA: {len(incomplete_final)} palabras con datos incompletos:")
        print(f"   {incomplete_final}...")
        print(f"   Serán reprocesadas en la próxima ejecución.")
    else:
        print(f"\nÉXITO: Todas las {len(final_df)} palabras procesadas correctamente.")
    
    print(f"\nResumen final:")
    print(f"   Total de palabras: {len(final_df)}")
    if incomplete_final:
        complete_words = len(final_df) - len(incomplete_final)
        print(f"   Completas: {complete_words}")
        print(f"   Incompletas: {len(incomplete_final)}")
    
    return final_df


def _generate_single_response(model, tokenizer, messages: List[Dict[str, str]], device: str, top_k: int = 10, do_sample: bool = False) -> Dict[str, Any]:
    """
    Generate responses.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")
        
        # Prepare input tokens
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            [text], return_tensors="pt", add_special_tokens=False
        ).to(device)
        
        # Get pre-inference probability distribution for first token
        with torch.no_grad():
            outputs_for_logits = model(**inputs)
            next_token_logits = outputs_for_logits.logits[0, -1, :]
            next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
            topk_list = _build_topk_list(next_token_logprobs, top_k, tokenizer)
        
        # Generate actual response
        outputs = model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=32768,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode generated text
        input_len = inputs["input_ids"].size(1)
        generated_text = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        
        # Return result with pre-inference probabilities for logprob calculation
        result_tokens = [{
            "step": 0,
            "token_id": None,
            "token": None,
            "logprob": None,
            "prob": None,
            "topk": topk_list  # Pre-inference distribution for digit scoring
        }]
        
        return {
            "prompt_text": text,
            "generated_text": generated_text,
            "tokens": result_tokens
        }