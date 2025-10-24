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
    # Load .env from the adapter_fusion directory
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai not installed. Some evaluation features may be limited.")
    OPENAI_AVAILABLE = False
    openai = None

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
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available, Cerebras client will not be initialized")
            return
        api_key = os.getenv('CEREBRAS_API_KEY')
        if not api_key:
            logger.warning("CEREBRAS_API_KEY not found in environment variables")
            return
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
            # GRPO format: prompt/answer
            if 'prompt' in item and 'answer' in item:
                qa_pairs.append({
                    'question': item['prompt'],
                    'answer': item['answer']
                })
            # Standard format: question/answer
            elif 'question' in item and 'answer' in item:
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
            elif 'messages' in item and isinstance(item['messages'], list):
                # Chat format with system/user/assistant messages
                user_message = None
                assistant_message = None

                for message in item['messages']:
                    if message.get('role') == 'user':
                        user_message = message.get('content', '')
                    elif message.get('role') == 'assistant':
                        assistant_message = message.get('content', '')

                if user_message and assistant_message:
                    qa_pairs.append({
                        'question': user_message,
                        'answer': assistant_message
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
        import time
        start_time = time.time()
        logger.info(f"🟢 STARTING MODEL INFERENCE - Loading model and generating response...")
        
        python_path = '/opt/anaconda3/bin/python'
        
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
                elapsed = time.time() - start_time
                logger.error(f"❌ MODEL INFERENCE FAILED in {elapsed:.3f}s - Return code: {process.returncode}")
                logger.error(f"STDERR: {process.stderr[:500]}")
                return f"Error: {process.stderr}"
            
            # Extract response
            output = process.stdout
            if "RESPONSE_START" in output and "RESPONSE_END" in output:
                start_idx = output.find("RESPONSE_START") + len("RESPONSE_START")
                end_idx = output.find("RESPONSE_END")
                response = output[start_idx:end_idx].strip()
            else:
                response = output.strip()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ MODEL INFERENCE COMPLETE in {elapsed:.3f} seconds - Response: {len(response)} chars")
            
            return response
        
        except subprocess.TimeoutExpired:
            return "Error: Timeout"
        except Exception as e:
            return f"Error: {e}"
    
    def evaluate_response(self, question: str, training_answer: str, model_response: str) -> Dict:
        """Evaluate response using Cerebras (ultra-fast)."""
        
        # Check if cerebras client is available
        if not self.cerebras_client:
            logger.warning("Cerebras client not available, returning default scores")
            return {
                "faithfulness": 5,
                "fact_recall": 5,
                "consistency": 5,
                "hallucination": 5,
                "facts_recalled": [],
                "facts_missing": [],
                "facts_added": [],
                "explanation": "Evaluation unavailable: Cerebras client not initialized (OpenAI library or API key missing)"
            }
        
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
            import time
            start_time = time.time()
            logger.info(f"🔵 CALLING CEREBRAS API - Starting request...")
            
            response = self.cerebras_client.chat.completions.create(
                model="llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ CEREBRAS API RESPONDED in {elapsed:.3f} seconds")
            
            response_text = response.choices[0].message.content
            logger.info(f"📊 Response length: {len(response_text)} chars")
            
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
    
    def compare_base_vs_adapter(self, adapter_name: str, training_data_path: str, num_questions: int = 20, progress_callback=None) -> Dict:
        """Compare base model vs adapter on the same questions."""
        logger.info(f"Starting comparison: base model vs adapter {adapter_name}")
        
        # Load adapter config
        config = self.load_adapter_config(adapter_name)
        model_path = config.get('model', '')
        adapter_path = os.path.join(self.adapter_base_dir, adapter_name)
        
        # Load training data
        training_data = self.load_training_data(training_data_path)
        qa_pairs = self.extract_qa_pairs(training_data)
        
        # Use Cerebras to generate questions from training data
        logger.info("Generating evaluation questions using Cerebras...")
        questions = self._generate_questions_with_cerebras(qa_pairs, num_questions)
        
        # Evaluate each question with both models
        comparisons = []
        for i, question_data in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}")
            
            if progress_callback:
                progress = int((i / len(questions)) * 100)
                progress_callback(i, len(questions), progress)
            
            question = question_data['question']
            expected_answer = question_data['expected_answer']
            
            # Get response from base model
            base_response = self.generate_response(None, model_path, question)
            
            # Get response from adapter
            adapter_response = self.generate_response(adapter_path, model_path, question)
            
            # Evaluate both responses
            base_eval = self.evaluate_response(question, expected_answer, base_response)
            adapter_eval = self.evaluate_response(question, expected_answer, adapter_response)
            
            comparisons.append({
                'question': question,
                'expected_answer': expected_answer,
                'base_response': base_response,
                'adapter_response': adapter_response,
                'base_scores': base_eval,
                'adapter_scores': adapter_eval
            })
        
        # Calculate aggregate scores
        base_avg = {
            'faithfulness': sum(c['base_scores']['faithfulness'] for c in comparisons) / len(comparisons),
            'fact_recall': sum(c['base_scores']['fact_recall'] for c in comparisons) / len(comparisons),
            'consistency': sum(c['base_scores']['consistency'] for c in comparisons) / len(comparisons),
            'hallucination': sum(c['base_scores']['hallucination'] for c in comparisons) / len(comparisons)
        }
        
        adapter_avg = {
            'faithfulness': sum(c['adapter_scores']['faithfulness'] for c in comparisons) / len(comparisons),
            'fact_recall': sum(c['adapter_scores']['fact_recall'] for c in comparisons) / len(comparisons),
            'consistency': sum(c['adapter_scores']['consistency'] for c in comparisons) / len(comparisons),
            'hallucination': sum(c['adapter_scores']['hallucination'] for c in comparisons) / len(comparisons)
        }
        
        base_overall = sum(base_avg.values()) / 4
        adapter_overall = sum(adapter_avg.values()) / 4
        
        report = {
            'adapter_name': adapter_name,
            'adapter_config': config,
            'training_data_path': training_data_path,
            'num_questions': len(questions),
            'evaluation_date': datetime.now().isoformat(),
            'base_model_scores': {
                'overall': round(base_overall * 10, 1),
                'faithfulness': round(base_avg['faithfulness'] * 10, 1),
                'fact_recall': round(base_avg['fact_recall'] * 10, 1),
                'consistency': round(base_avg['consistency'] * 10, 1),
                'hallucination': round(base_avg['hallucination'] * 10, 1)
            },
            'adapter_scores': {
                'overall': round(adapter_overall * 10, 1),
                'faithfulness': round(adapter_avg['faithfulness'] * 10, 1),
                'fact_recall': round(adapter_avg['fact_recall'] * 10, 1),
                'consistency': round(adapter_avg['consistency'] * 10, 1),
                'hallucination': round(adapter_avg['hallucination'] * 10, 1)
            },
            'detailed_comparisons': comparisons
        }
        
        logger.info(f"Comparison complete. Base: {report['base_model_scores']['overall']}/100, Adapter: {report['adapter_scores']['overall']}/100")
        return report
    
    def _generate_questions_with_cerebras(self, qa_pairs: List[Dict[str, str]], num_questions: int) -> List[Dict[str, str]]:
        """Use Cerebras to generate evaluation questions from training data."""
        if not self.cerebras_client:
            logger.warning("Cerebras not available, using random selection from training data")
            selected = random.sample(qa_pairs, min(num_questions, len(qa_pairs)))
            return [{'question': qa['question'], 'expected_answer': qa['answer']} for qa in selected]
        
        # Sample training examples to give Cerebras context
        sample_size = min(10, len(qa_pairs))
        samples = random.sample(qa_pairs, sample_size)
        
        context = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in samples])
        
        prompt = f"""Based on this training data, generate {num_questions} diverse evaluation questions that test understanding of the material.

Training data samples:
{context}

Generate {num_questions} questions in JSON format:
[
  {{"question": "...", "expected_answer": "..."}},
  ...
]

Make questions diverse and challenging. Expected answers should be concise."""

        try:
            import time
            start_time = time.time()
            logger.info("🔵 Generating questions with Cerebras...")
            
            response = self.cerebras_client.chat.completions.create(
                model="llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": "You are an expert at creating evaluation questions. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Questions generated in {elapsed:.3f} seconds")
            
            response_text = response.choices[0].message.content
            
            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            questions = json.loads(response_text)
            logger.info(f"Generated {len(questions)} questions")
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Failed to generate questions with Cerebras: {e}")
            logger.warning("Falling back to random selection from training data")
            selected = random.sample(qa_pairs, min(num_questions, len(qa_pairs)))
            return [{'question': qa['question'], 'expected_answer': qa['answer']} for qa in selected]
    
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
        if len(results) == 0:
            raise ValueError("No evaluation results generated. Check that the training data format is correct and contains Q&A pairs.")

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
    
    def save_comparison_report(self, report: Dict, output_dir: str = "./evaluation_results"):
        """Save comparison report with detailed Q&A."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON with all questions and answers
        json_file = os.path.join(output_dir, f"{report['adapter_name']}_comparison_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable comparison
        txt_file = os.path.join(output_dir, f"{report['adapter_name']}_comparison_{timestamp}.txt")
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BASE MODEL vs ADAPTER COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Adapter: {report['adapter_name']}\n")
            f.write(f"Date: {report['evaluation_date']}\n")
            f.write(f"Questions: {report['num_questions']}\n\n")
            
            f.write("OVERALL SCORES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Base Model:  {report['base_model_scores']['overall']}/100\n")
            f.write(f"Adapter:     {report['adapter_scores']['overall']}/100\n")
            f.write(f"Improvement: {report['adapter_scores']['overall'] - report['base_model_scores']['overall']:+.1f}\n\n")
            
            f.write("DETAILED SCORES:\n")
            f.write("-" * 80 + "\n")
            for metric in ['faithfulness', 'fact_recall', 'consistency', 'hallucination']:
                base_score = report['base_model_scores'][metric]
                adapter_score = report['adapter_scores'][metric]
                diff = adapter_score - base_score
                f.write(f"{metric.title():15} Base: {base_score:5.1f}  Adapter: {adapter_score:5.1f}  Diff: {diff:+5.1f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("QUESTION-BY-QUESTION COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            for i, comp in enumerate(report['detailed_comparisons'], 1):
                f.write(f"\nQUESTION {i}:\n")
                f.write(f"{comp['question']}\n\n")
                
                f.write(f"EXPECTED ANSWER:\n{comp['expected_answer']}\n\n")
                
                f.write(f"BASE MODEL RESPONSE:\n{comp['base_response']}\n")
                f.write(f"Scores: F={comp['base_scores']['faithfulness']} R={comp['base_scores']['fact_recall']} C={comp['base_scores']['consistency']} H={comp['base_scores']['hallucination']}\n\n")
                
                f.write(f"ADAPTER RESPONSE:\n{comp['adapter_response']}\n")
                f.write(f"Scores: F={comp['adapter_scores']['faithfulness']} R={comp['adapter_scores']['fact_recall']} C={comp['adapter_scores']['consistency']} H={comp['adapter_scores']['hallucination']}\n")
                f.write("-" * 80 + "\n")
        
        logger.info(f"Comparison report saved to {output_dir}")
        logger.info(f"JSON: {json_file}")
        logger.info(f"Text: {txt_file}")
        return json_file, txt_file
    
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
            f.write(f"  Overall: {report['scores']['overall']}/100\n")
            f.write(f"  Faithfulness: {report['scores']['faithfulness']}/100\n")
            f.write(f"  Fact Recall: {report['scores']['fact_recall']}/100\n")
            f.write(f"  Consistency: {report['scores']['consistency']}/100\n")
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
