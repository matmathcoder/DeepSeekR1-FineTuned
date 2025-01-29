from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PPOTrainer, PPOConfig
import os
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

# Load the pre-trained DeepSeek model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-1.5b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom dataset class for Django repositories
class DjangoCodeDataset(Dataset):
    def __init__(self, data_dir="django_data"):
        self.file_paths = []
        # Define patterns for important Django files
        important_patterns = [
            'views.py',
            'models.py',
            'urls.py',
            'forms.py',
            'admin.py',
            'middleware.py',
            'serializers.py'
        ]
        
        # Define patterns to exclude
        exclude_patterns = [
            'migrations/',
            'tests/',
            'test_',
            'venv/',
            '__pycache__/',
            '.git/',
            'static/',
            'media/',
            'templates/'
        ]
        
        for root, dirs, files in os.walk(data_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                # Only include important Django files
                if any(pattern in file for pattern in important_patterns):
                    file_path = os.path.join(root, file)
                    # Basic size check to exclude very large files
                    if os.path.getsize(file_path) < 500000:  # 500KB limit
                        self.file_paths.append(file_path)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                file_name = os.path.basename(self.file_paths[idx])
                file_type = file_name.replace('.py', '')
                
                # Create more specific prompts based on file type
                prompts = {
                    'views': 'Write a Django view for: ',
                    'models': 'Create a Django model for: ',
                    'urls': 'Define URL patterns for: ',
                    'forms': 'Create a Django form for: ',
                    'admin': 'Write Django admin configuration for: ',
                    'middleware': 'Create Django middleware for: ',
                    'serializers': 'Write a Django REST framework serializer for: '
                }
                
                base_prompt = prompts.get(file_type, 'Write Django code for: ')
                context = self._extract_context(content)
                prompt = f"{base_prompt}{context}"
                
                return {
                    "input_texts": prompt,
                    "reference_texts": content
                }
            except UnicodeDecodeError:
                return {
                    "input_texts": "",
                    "reference_texts": ""
                }
    
    def _extract_context(self, content):
        """Extract meaningful context from the file content"""
        # Get first docstring or first few lines
        lines = content.split('\n')
        context = []
        for line in lines[:10]:  # Look at first 10 lines
            if line.strip().startswith('class ') or line.strip().startswith('def '):
                context.append(line.strip())
        return ' '.join(context) if context else "Generic Django component"

# Initialize dataset and dataloader
dataset = DjangoCodeDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define the PPO configuration
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=2,  # Reduced batch size for memory
    gradient_accumulation_steps=4,  # Accumulate gradients
    optimize_cuda_cache=True,
    log_with="tensorboard",
    max_grad_norm=0.5,
    early_stopping=True,
    target_kl=0.1
)

# Initialize the PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
)

def reward_function(generated_text, reference_text):
    # Simple similarity-based reward
    # You might want to implement more sophisticated metrics
    generated_tokens = set(generated_text.split())
    reference_tokens = set(reference_text.split())
    overlap = len(generated_tokens.intersection(reference_tokens))
    total = len(generated_tokens.union(reference_tokens))
    return overlap / total if total > 0 else 0

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for batch in dataloader:
        # Skip empty examples
        if not batch["input_texts"] or not batch["reference_texts"]:
            continue
            
        # Tokenize input and reference texts
        inputs = tokenizer(batch["input_texts"], return_tensors="pt", padding=True, truncation=True)
        references = batch["reference_texts"]

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1
            )

        # Decode generated texts
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Compute rewards
        rewards = [reward_function(gen, ref) for gen, ref in zip(generated_texts, references)]

        # Perform PPO step
        ppo_trainer.step(inputs, outputs, rewards)

    print(f"Completed epoch {epoch + 1}/{num_epochs}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_deepseek_django")
tokenizer.save_pretrained("fine_tuned_deepseek_django")
