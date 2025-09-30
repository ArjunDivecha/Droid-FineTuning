#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: evaluate_adapters.py
=============================================================================

INPUT FILES:
- Training data JSONL file (Q&A pairs)
- Adapter directories with adapter_config.json and adapters.safetensors

OUTPUT FILES:
- evaluation_report.json: Detailed scores and analysis
- evaluation_summary.txt: Human-readable summary

VERSION: 1.0
LAST UPDATED: 2025-01-29
AUTHOR: Droid-FineTuning Project

DESCRIPTION:
Evaluates adapter faithfulness to training data using Claude Sonnet 4.5.
Measures fact recall, consistency, and hallucination rates.

DEPENDENCIES:
- anthropic
- python-dotenv
- mlx_lm
- json

USAGE:
python evaluate_adapters.py --adapter mlx_finetune --training-data train.jsonl

NOTES:
- Requires ANTHROPIC_API_KEY in .env file
- Uses Claude Sonnet 4.5 for evaluation
- Evaluates 20 test questions by default
=============================================================================
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import random

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

try:
    import openai
except ImportError:
    print("Error: openai not installed. Install with: pip install openai")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdapterEvaluator:
    """Evaluates adapter faithfulness to training data."""
    
    def __init__(self, adapter_base_dir: str = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"):
        self.adapter_base_dir = adapter_base_dir
        self.cerebras_client = None
        self._init_cerebras()
        
    def _init_cerebras(self):
        """Initialize Cerebras client."""
        api_key = os.getenv('CEREBRAS_API_KEY')
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not found in environment variables")
        self.cerebras_client = openai.OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key
        )
        logger.info("Cerebras client initialized")
    
    def load_adapter_config(self, adapter_name: str) -> Dict:
        """Load adapter configuration."""
        adapter_dir = os.path.join(self.adapter_base_dir, adapter_name)
        config_file = os.path.join(adapter_dir, "adapter_config.json")
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"No config found at {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded config for adapter: {adapter_name}")
        return config
    
    def load_training_data(self, data_path: str) -> List[Dict]:
        """Load training data from JSONL file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} examples from training data")
        return data
    
    def extract_qa_pairs(self, training_data: List[Dict]) -> List[Dict[str, str]]:
        """Extract Q&A pairs from training data."""
        qa_pairs = []
        
        for item in training_data:
            # Handle different JSONL formats
            if 'question' in item and 'answer' in item:
                qa_pairs.append({
                    'question': item['question'],
                    'answer': item['answer']
                })
            elif 'messages' in item and 'completions' in item:
                # RFT format with messages/completions/ranking
                question = item['messages'][0]['content']
                # Use the best completion (lowest ranking index)
                ranking = item.get('ranking', [0])
                best_idx = ranking[0] if ranking else 0
                answer = item['completions'][best_idx]['text']
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })
            elif 'text' in item:
                # Parse chat format or Q&A format
                text = item['text']
                
                # Try to extract from chat format (user/assistant)
                if '<|im_start|>user' in text and '<|im_end|>' in text:
                    # Extract user message
                    user_start = text.find('<|im_start|>user') + len('<|im_start|>user')
                    user_end = text.find('<|im_end|>', user_start)
                    if user_start > 0 and user_end > user_start:
                        question = text[user_start:user_end].strip()
                        
                        # Extract assistant message if present
                        assistant_start = text.find('<|im_start|>assistant', user_end)
                        if assistant_start > 0:
                            assistant_start += len('<|im_start|>assistant')
                            assistant_end = text.find('<|im_end|>', assistant_start)
                            if assistant_end > assistant_start:
                                answer = text[assistant_start:assistant_end].strip()
                            else:
                                answer = "No answer provided in training"
                        else:
                            # No assistant response, use question as both
                            answer = "Training prompt only"
                        
                        qa_pairs.append({
                            'question': question,
                            'answer': answer
                        })
                # Try Q: A: format
                elif 'Q:' in text or 'Question:' in text:
                    parts = text.split('\n')
                    question = None
                    answer = None
                    for part in parts:
                        if part.startswith('Q:') or part.startswith('Question:'):
                            question = part.split(':', 1)[1].strip()
                        elif part.startswith('A:') or part.startswith('Answer:') or part.startswith('Assistant:'):
                            answer = part.split(':', 1)[1].strip()
                    
                    if question and answer:
                        qa_pairs.append({
                            'question': question,
                            'answer': answer
                        })
        
        logger.info(f"Extracted {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def select_test_questions(self, qa_pairs: List[Dict[str, str]], num_questions: int = 20) -> List[Dict[str, str]]:
        """Select test questions from Q&A pairs."""
        if len(qa_pairs) <= num_questions:
            return qa_pairs
        
        # Random sample
        test_set = random.sample(qa_pairs, num_questions)
        logger.info(f"Selected {num_questions} test questions")
        return test_set
    
    def generate_response(self, adapter_path: Optional[str], model_path: str, prompt: str) -> str:
        """Generate response from adapter or base model using MLX."""
        python_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python'
        
        # Build adapter_path parameter
        adapter_param = f'adapter_path="{adapter_path}"' if adapter_path else 'adapter_path=None'
        
        cmd = [
            python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

try:
    model, tokenizer = load("{model_path}", {adapter_param})
    prompt = """{prompt}"""
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
    print("RESPONSE_START")
    print(response)
    print("RESPONSE_END")
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
'''
        ]
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if process.returncode != 0:
                return f"Error: {process.stderr}"
            
            # Extract response
            output = process.stdout
            if "RESPONSE_START" in output and "RESPONSE_END" in output:
                start_idx = output.find("RESPONSE_START") + len("RESPONSE_START")
                end_idx = output.find("RESPONSE_END")
                response = output[start_idx:end_idx].strip()
            else:
                response = output.strip()
            
            return response
        
        except subprocess.TimeoutExpired:
            return "Error: Timeout"
        except Exception as e:
            return f"Error: {e}"
    
    def evaluate_response(self, question: str, training_answer: str, model_response: str) -> Dict:
        """Evaluate response using Cerebras (ultra-fast)."""
        
        prompt = f"""You are evaluating if a model's response is faithful to its training data.

QUESTION:
{question}

TRAINING DATA ANSWER (Ground Truth):
{training_answer}

MODEL RESPONSE:
{model_response}

Evaluate the model's response on these criteria:

1. FAITHFULNESS (1-10): How well does the response match the training data?
   - 10 = Perfect match, all key facts from training present
   - 5 = Partial match, some facts present
   - 1 = No match, different information

2. FACT RECALL (1-10): Did it include the key facts from training?
   - 10 = All key facts recalled
   - 5 = Some facts recalled
   - 1 = Few or no facts recalled

3. CONSISTENCY (1-10): Does it contradict the training data?
   - 10 = Fully consistent, no contradictions
   - 5 = Minor inconsistencies
   - 1 = Major contradictions

4. HALLUCINATION (1-10): Does it add information NOT in training data?
   - 10 = Only uses trained information
   - 5 = Some additions, mostly accurate
   - 1 = Mostly made-up information

Provide your evaluation in JSON format:
{{
  "faithfulness": <score 1-10>,
  "fact_recall": <score 1-10>,
  "consistency": <score 1-10>,
  "hallucination": <score 1-10>,
  "facts_recalled": ["fact1", "fact2", ...],
  "facts_missing": ["fact1", "fact2", ...],
  "facts_added": ["fact1", "fact2", ...],
  "explanation": "Brief explanation of scores"
}}"""

        try:
            response = self.cerebras_client.chat.completions.create(
                model="llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            evaluation = json.loads(response_text)
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "faithfulness": 0,
                "fact_recall": 0,
                "consistency": 0,
                "hallucination": 0,
                "facts_recalled": [],
                "facts_missing": [],
                "facts_added": [],
                "explanation": f"Evaluation error: {str(e)}"
            }
    
    def evaluate_adapter(self, adapter_name: str, training_data_path: str, num_questions: int = 20, progress_callback=None, use_base_model: bool = False) -> Dict:
        """Evaluate an adapter against training data."""
        eval_type = "base model" if use_base_model else "adapter"
        logger.info(f"Starting evaluation of {eval_type}: {adapter_name}")
        
        # Load adapter config
        config = self.load_adapter_config(adapter_name)
        model_path = config.get('model', '')
        adapter_path = None if use_base_model else os.path.join(self.adapter_base_dir, adapter_name)
        
        # Load training data
        training_data = self.load_training_data(training_data_path)
        qa_pairs = self.extract_qa_pairs(training_data)
        test_set = self.select_test_questions(qa_pairs, num_questions)
        
        # Evaluate each question
        results = []
        for i, qa in enumerate(test_set, 1):
            logger.info(f"Evaluating question {i}/{len(test_set)}")
            
            # Update progress if callback provided
            if progress_callback:
                progress = int((i / len(test_set)) * 100)
                progress_callback(i, len(test_set), progress)
            
            question = qa['question']
            training_answer = qa['answer']
            
            # Generate response
            model_response = self.generate_response(adapter_path, model_path, question)
            
            # Evaluate response
            evaluation = self.evaluate_response(question, training_answer, model_response)
            
            results.append({
                'question': question,
                'training_answer': training_answer,
                'model_response': model_response,
                'evaluation': evaluation
            })
        
        # Calculate aggregate scores
        avg_faithfulness = sum(r['evaluation']['faithfulness'] for r in results) / len(results)
        avg_fact_recall = sum(r['evaluation']['fact_recall'] for r in results) / len(results)
        avg_consistency = sum(r['evaluation']['consistency'] for r in results) / len(results)
        avg_hallucination = sum(r['evaluation']['hallucination'] for r in results) / len(results)
        
        overall_score = (avg_faithfulness + avg_fact_recall + avg_consistency + avg_hallucination) / 4
        
        report = {
            'adapter_name': adapter_name,
            'is_base_model': use_base_model,
            'adapter_config': config,
            'training_data_path': training_data_path,
            'num_questions': len(test_set),
            'evaluation_date': datetime.now().isoformat(),
            'scores': {
                'overall': round(overall_score * 10, 1),
                'faithfulness': round(avg_faithfulness * 10, 1),
                'fact_recall': round(avg_fact_recall * 10, 1),
                'consistency': round(avg_consistency * 10, 1),
                'hallucination': round(avg_hallucination * 10, 1)
            },
            'detailed_results': results
        }
        
        logger.info(f"Evaluation complete. Overall score: {report['scores']['overall']}/100")
        return report
    
    def save_report(self, report: Dict, output_dir: str = "./evaluation_results"):
        """Save evaluation report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        json_file = os.path.join(output_dir, f"{report['adapter_name']}_evaluation.json")
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save summary
        summary_file = os.path.join(output_dir, f"{report['adapter_name']}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ADAPTER EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Adapter: {report['adapter_name']}\n")
            f.write(f"Evaluated: {report['evaluation_date']}\n")
            f.write(f"Test Questions: {report['num_questions']}\n\n")
            f.write("SCORES:\n")
            f.write(f"  Overall:       {report['scores']['overall']}/100\n")
            f.write(f"  Faithfulness:  {report['scores']['faithfulness']}/100\n")
            f.write(f"  Fact Recall:   {report['scores']['fact_recall']}/100\n")
            f.write(f"  Consistency:   {report['scores']['consistency']}/100\n")
            f.write(f"  Hallucination: {report['scores']['hallucination']}/100\n\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Report saved to {output_dir}")
        return json_file, summary_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate adapter faithfulness to training data")
    parser.add_argument("--adapter", required=True, help="Adapter name to evaluate")
    parser.add_argument("--training-data", help="Path to training data JSONL file")
    parser.add_argument("--num-questions", type=int, default=20, help="Number of test questions")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    evaluator = AdapterEvaluator()
    
    # Get training data path
    if args.training_data:
        training_data_path = args.training_data
    else:
        # Try to get from adapter config
        try:
            config = evaluator.load_adapter_config(args.adapter)
            training_data_path = config.get('data', '')
            if training_data_path and os.path.isdir(training_data_path):
                # Look for JSONL files in directory
                jsonl_files = list(Path(training_data_path).glob("*.jsonl"))
                if jsonl_files:
                    training_data_path = str(jsonl_files[0])
            
            if not training_data_path or not os.path.exists(training_data_path):
                training_data_path = input("Enter path to training data JSONL file: ")
        except Exception as e:
            training_data_path = input("Enter path to training data JSONL file: ")
    
    # Run evaluation
    report = evaluator.evaluate_adapter(args.adapter, training_data_path, args.num_questions)
    
    # Save report
    json_file, summary_file = evaluator.save_report(report, args.output_dir)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Overall Score: {report['scores']['overall']}/100")
    print(f"Report saved to: {json_file}")

if __name__ == "__main__":
    main()
