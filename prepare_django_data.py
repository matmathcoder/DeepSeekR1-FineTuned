import os
import json
import ast
import logging
from pathlib import Path
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DjangoCodeExtractor:
    def __init__(self, input_dir, output_file):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.django_patterns = [
            r'from django\..*? import .*?$',
            r'class .*?\(.*?View.*?\):',
            r'class .*?\(.*?Model.*?\):',
            r'class .*?\(.*?Form.*?\):',
            r'class .*?\(.*?Admin.*?\):',
            r'@.*?decorator',
            r'urlpatterns.*?=.*?\[',
        ]
    
    def is_django_file(self, content):
        return any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in self.django_patterns
        )
    
    def extract_code_with_context(self, file_path):
        """Extract Django code patterns with context."""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not self.is_django_file(content):
                return []
            
            # Extract Django classes and functions
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Process Django classes
                    bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
                    if any(base.endswith(('Model', 'View', 'Form', 'Admin')) for base in bases):
                        class_type = next(base for base in bases if base.endswith(('Model', 'View', 'Form', 'Admin')))
                        class_lines = content.split('\n')[node.lineno-1:node.end_lineno]
                        class_code = '\n'.join(class_lines)
                        
                        examples.append({
                            "instruction": f"Create a Django {class_type} class with the following requirements",
                            "input": f"Create a {class_type} named '{node.name}' with appropriate methods and attributes.",
                            "output": class_code
                        })
                
                elif isinstance(node, ast.FunctionDef):
                    # Process Django view functions
                    if any(dec.id in ['login_required', 'permission_required'] for dec in node.decorator_list if isinstance(dec, ast.Name)):
                        func_lines = content.split('\n')[node.lineno-1:node.end_lineno]
                        func_code = '\n'.join(func_lines)
                        
                        examples.append({
                            "instruction": "Create a Django view function with the following requirements",
                            "input": f"Create a view function named '{node.name}' with appropriate decorators and logic.",
                            "output": func_code
                        })
        
        except Exception as e:
            logging.warning(f"Error processing {file_path}: {str(e)}")
            return []
        
        return examples
    
    def process_directory(self):
        """Process all Python files and create dataset."""
        all_examples = []
        python_files = list(self.input_dir.rglob("*.py"))
        
        logging.info(f"Found {len(python_files)} Python files")
        
        for file_path in tqdm(python_files, desc="Processing files"):
            examples = self.extract_code_with_context(file_path)
            all_examples.extend(examples)
        
        # Remove duplicates based on output
        unique_examples = {ex['output']: ex for ex in all_examples}.values()
        
        logging.info(f"Generated {len(unique_examples)} unique training examples")
        
        # Save to JSON in LLaMA-Factory format
        output_data = list(unique_examples)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Training data saved to {self.output_file}")

def main():
    input_dir = "django_data"
    output_file = "LLaMA-Factory/data/django_code.json"
    
    extractor = DjangoCodeExtractor(input_dir, output_file)
    extractor.process_directory()

if __name__ == "__main__":
    main()
