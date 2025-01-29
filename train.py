import os
import gc
import logging
import random
from pathlib import Path
from datetime import datetime
import shutil
from tqdm import tqdm
import requests
from llama_cpp import Llama
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_model(url, path):
    if path.exists():
        logging.info(f"Model already exists at {path}")
        return
    
    logging.info(f"Downloading model from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for data in r.iter_content(chunk_size=8192):
                    f.write(data)
                    pbar.update(len(data))

class LlamaCppModel:
    def __init__(self, model_path):
        try:
            logging.info("Initializing model with optimized settings...")
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=256,            # Minimal context
                n_threads=2,          # Reduced threads
                n_batch=1,           # Single batch
                verbose=True,        # Debug mode
                seed=42,
                n_gpu_layers=0,      # CPU only
                use_mmap=True,       # Enable memory mapping
                use_mlock=False,     # Don't lock memory
                logits_all=False,    # Don't compute all logits
                embedding=False,     # Don't compute embeddings
                chat_format="llama-2"  # Explicit chat format
            )
            # Test model with a simple completion
            test_output = self.llm.create_completion(
                prompt="print('hello')",
                max_tokens=10,
                temperature=0.0,
                echo=False
            )
            logging.info("Model test successful")
            logging.info(f"Test output: {test_output['choices'][0]['text'] if test_output and 'choices' in test_output else 'No output'}")
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

def prepare_dataset(max_files=100, max_examples=500):
    dataset = []
    django_files = []
    django_data_dir = "django_data"
    
    # Collect Python files
    for root, _, files in os.walk(django_data_dir):
        for file in files:
            if file.endswith('.py'):
                django_files.append(os.path.join(root, file))
                if len(django_files) >= max_files:
                    break
        if len(django_files) >= max_files:
            break
    
    logging.info(f"Processing {len(django_files)} Django files")
    
    # Process files
    for file_path in tqdm(django_files, desc="Preparing dataset"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple extraction of functions and classes
            lines = content.split('\n')
            current_block = []
            
            for line in lines:
                if line.strip().startswith(('def ', 'class ')):
                    if current_block:
                        block = '\n'.join(current_block)
                        if len(block.strip()) > 0:
                            dataset.append({
                                "input": f"Generate Django code:\n{block}\n",
                                "output": block
                            })
                    current_block = [line]
                elif current_block:
                    current_block.append(line)
            
            # Add last block
            if current_block:
                block = '\n'.join(current_block)
                if len(block.strip()) > 0:
                    dataset.append({
                        "input": f"Generate Django code:\n{block}\n",
                        "output": block
                    })
            
            if len(dataset) >= max_examples:
                break
                
        except Exception as e:
            logging.warning(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Limit dataset size
    if len(dataset) > max_examples:
        dataset = random.sample(dataset, max_examples)
    
    logging.info(f"Dataset prepared with {len(dataset)} examples")
    return dataset

def train_model():
    # Set up paths
    model_url = "https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_K_M.gguf"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "codellama-7b.Q4_K_M.gguf"
    
    # Download model if needed
    try:
        download_model(model_url, model_path)
    except Exception as e:
        logging.error(f"Failed to download model: {str(e)}")
        raise
    
    # Initialize model with minimal settings
    try:
        model = LlamaCppModel(model_path)
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise
    
    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            use_fast=False
        )
        logging.info("Tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {str(e)}")
        raise
    
    # Prepare minimal dataset
    logging.info("Preparing dataset...")
    train_dataset = prepare_dataset(max_files=100, max_examples=500)
    
    # Training configuration
    num_epochs = 1  # Reduced to 1 epoch for testing
    checkpoint_interval = 50
    output_dir = Path(f"fine_tuned_codellama_django_{datetime.now().strftime('%Y%m%d_%H%M')}")
    output_dir.mkdir(exist_ok=True)
    
    try:
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, example in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                if batch_idx % 5 == 0:
                    gc.collect()
                
                try:
                    # Basic completion with minimal parameters
                    completion = model.llm.create_completion(
                        prompt=example['input'],
                        max_tokens=32,  # Very small for testing
                        temperature=0.0,  # Deterministic
                        stop=["</s>", "\n\n"],  # Stop at end of generation
                        echo=False,  # Don't include prompt in output
                    )
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        logging.info(f"Processed batch {batch_idx}")
                        if completion and 'choices' in completion and len(completion['choices']) > 0:
                            logging.info(f"Sample output: {completion['choices'][0]['text'][:50]}...")
                    
                    # Save checkpoint less frequently
                    if batch_idx > 0 and batch_idx % checkpoint_interval == 0:
                        checkpoint_path = output_dir / f"checkpoint_epoch{epoch}_batch{batch_idx}.gguf"
                        shutil.copy2(model_path, checkpoint_path)
                        logging.info(f"Saved checkpoint at batch {batch_idx}")
                
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            logging.info(f"Completed epoch {epoch + 1}")
            gc.collect()
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    finally:
        # Save final model
        final_model_path = output_dir / "final_model.gguf"
        try:
            shutil.copy2(model_path, final_model_path)
            logging.info(f"Final model saved: {final_model_path}")
        except Exception as e:
            logging.error(f"Failed to save final model: {str(e)}")

if __name__ == "__main__":
    train_model()