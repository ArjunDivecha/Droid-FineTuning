#!/usr/bin/env python3
"""
=============================================================================
VOICE AUTHENTICITY TEST WITH LOCAL LM STUDIO MODELS
=============================================================================

VERSION: 2.0
LAST UPDATED: 2025-01-22
AUTHOR: Arjun B. Divecha

Includes both OpenAI fine-tuned models AND local LM Studio models in the evaluation.

PURPOSE:
This program comprehensively tests and compares voice authenticity across multiple AI models,
including both OpenAI fine-tuned models and local LM Studio models. It evaluates how well
each model captures authentic communication patterns through scenario-based testing and
AI-powered evaluation.

INPUT FILES:
- /Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/New Writing Dataset/dataset_output/train_dataset.jsonl
  - Format: JSONL (JSON Lines) format with training data
  - Contains: Real voice examples from Arjun's communication patterns
  - Required: For loading authentic voice samples for evaluation context
  - Note: Only first 10 examples are used for efficiency

OUTPUT FILES:
- voice_test_with_local_[timestamp].json
  - Location: Current working directory
  - Format: JSON with indented formatting
  - Contents: Raw test responses from all models for each scenario
  - Includes: Model responses, metadata, timestamps, voice examples

- voice_evaluation_with_local_[timestamp].json
  - Location: Current working directory
  - Format: JSON with indented formatting
  - Contents: Claude Opus-4 evaluation of voice authenticity for all models
  - Includes: Authenticity scores, detailed analysis, rankings, local vs OpenAI comparisons

REQUIREMENTS:
- Python 3.8+
- LM Studio running locally with API server enabled (http://localhost:1234)
- Local models loaded in LM Studio (e.g., arjun16, arjun4q)
- Valid OpenAI API key
- Valid Anthropic API key
- Required packages: openai, anthropic, requests, pandas

USAGE:
1. Start LM Studio and enable local server (usually http://localhost:1234)
2. Load your local fine-tuned models in LM Studio
3. Ensure API keys are configured in the script
4. Run: python voice_test_with_local_models.py

TEST METHODOLOGY:
- 5 scenario-based tests matching real communication patterns
- Neutral system prompts to avoid biasing results
- Both OpenAI fine-tuned and local models tested
- Claude Opus-4 evaluation of voice authenticity (1-10 scale)
- Comparison between local and cloud-based fine-tuned models

VOICE CHARACTERISTICS EVALUATED:
- Casual, direct communication with contractions
- Conversational tone vs formal language
- Short, punchy sentences
- Practical focus and questions
- Informal terms when appropriate
- Personal touch vs corporate speak

VERSION HISTORY:
- v1.0 (2024-XX-XX): Initial implementation with OpenAI models only
- v2.0 (2025-01-22): Added local LM Studio model support, enhanced evaluation framework
=============================================================================
"""

import openai
import anthropic
import requests
import json
import pandas as pd
from datetime import datetime
import time
import os

# API Configuration
OPENAI_API_KEY = "sk-REDACTED_OPENAI_KEY"
ANTHROPIC_API_KEY = "sk-ant-REDACTED_ANTHROPIC_KEY"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"  # Default LM Studio API endpoint

# OpenAI Fine-tuned Models
# Dictionary mapping model names to their OpenAI model IDs
# These are various fine-tuned models trained on Arjun's voice/writing patterns
OPENAI_MODELS = {
    "myvoice_nano_val": "ft:gpt-4.1-nano-2025-04-14:personal:myvoice-val:C6S2j7pL",
    "new-writing": "ft:gpt-4o-2024-08-06:personal:arjun-voice-v2:C20wnbFD"
}

# Local LM Studio Models
# Dictionary mapping display names to model names as loaded in LM Studio
# These models should be running locally in LM Studio for testing
LOCAL_MODELS = {
"arjun16_local": "arjun16",  # Local fine-tuned model (16-bit quantization)
"arjun4q_local": "arjun4q",  # Local fine-tuned model (4-bit quantization)
"qwen3_next_80b_local": "qwen3-next-80b",  # Qwen 3 Next 80B model
    # Add more local models as needed:
# "another_local_model": "model_name_in_lm_studio",
}

# Test Scenarios
# List of realistic communication scenarios designed to test voice authenticity
# Each scenario includes the real-world context, formal version, and expected voice elements
# These scenarios are based on actual communication patterns from Arjun's writing samples
VOICE_TEST_SCENARIOS = [
    {
        "id": 1,
        "category": "Business Request",
        "scenario": "You need to ask someone for wire transfer instructions for a donation",
        "formal_version": "Could you please provide the wire transfer instructions for this year's charitable donation?",
        "expected_voice_elements": ["contractions", "direct_question", "casual_tone"]
    },
    {
        "id": 2, 
        "category": "Technical Issue",
        "scenario": "Your pool cleaner is broken and you need to report it",
        "formal_version": "The automated pool cleaning system appears to be malfunctioning and requires technical assistance.",
        "expected_voice_elements": ["casual_terms", "direct_description", "practical_request"]
    },
    {
        "id": 3,
        "category": "Service Update",
        "scenario": "You're telling someone you can now access your work systems again",
        "formal_version": "Please be advised that I have regained access to the corporate email and communication systems.",
        "expected_voice_elements": ["informal_greeting", "BTW_style", "normal_communication"]
    },
    {
        "id": 4,
        "category": "Clarification Request", 
        "scenario": "You need more details about fees and what happened",
        "formal_version": "Could you please provide additional clarification regarding the associated costs and the sequence of events?",
        "expected_voice_elements": ["short_questions", "direct_style", "want_specifics"]
    },
    {
        "id": 5,
        "category": "Business Decision",
        "scenario": "You're asking about corporate structure implications for payments",
        "formal_version": "If the agreement is structured under the S-corporation entity, how will this affect the consolidated reporting on the K1 tax document?",
        "expected_voice_elements": ["assume_scenario", "bundled_terminology", "practical_implications"]
    }
]

# System Prompt
# Neutral prompt used for all models to avoid biasing results toward any particular style
# Allows each model to respond in its trained/default communication style
NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant. Respond naturally to the user's request in your typical communication style."

class LocalModelTester:
    """
    Main testing class that handles both OpenAI and local LM Studio model testing.

    Manages API clients, coordinates testing across multiple models, and orchestrates
    the evaluation process using Claude for voice authenticity assessment.
    """
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.results = {"test_responses": {}, "metadata": {"timestamp": self.timestamp}}
        self.lm_studio_available = self.check_lm_studio()
    
    def check_lm_studio(self):
        """
        Check if LM Studio local server is running and accessible.

        Attempts to connect to LM Studio's API endpoint and verifies it's responding.
        Returns True if connection successful, False otherwise.
        """
        try:
            response = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"✅ LM Studio connected - {len(models.get('data', []))} models available")
                return True
            else:
                print(f"⚠️  LM Studio responding but with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ LM Studio not available: {e}")
            print("   Make sure LM Studio is running with local server enabled")
            return False
    
    def query_openai_model(self, model_id, scenario_description):
        """
    Query an OpenAI model with a scenario-based prompt.

    Args:
    model_id (str): OpenAI model identifier
    scenario_description (str): The scenario prompt for the model

    Returns:
    str: Model response or error message
    """
        try:
            prompt = f"Scenario: {scenario_description}\n\nHow would you communicate this? Write the message you would send:"
            
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"
    
    def query_local_model(self, model_name, scenario_description):
        """
    Query a local LM Studio model with a scenario-based prompt.

    Args:
    model_name (str): Model name as loaded in LM Studio
    scenario_description (str): The scenario prompt for the model

    Returns:
    str: Model response or error message
    """
        try:
            prompt = f"Scenario: {scenario_description}\n\nHow would you communicate this? Write the message you would send:"
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"ERROR: LM Studio returned status {response.status_code}"
                
        except Exception as e:
            return f"ERROR: {e}"
    
    def load_voice_examples(self):
        """Load sample voice examples from training data"""
        try:
            examples = []
            with open('/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/New Writing Dataset/dataset_output/train_dataset.jsonl', 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Just take first 10 examples
                        break
                    data = json.loads(line)
                    # Extract the assistant response (Arjun's voice)
                    for message in data['messages']:
                        if message['role'] == 'assistant':
                            examples.append(message['content'])
            return examples
        except Exception as e:
            print(f"Warning: Could not load voice examples: {e}")
            return []
    
    def run_voice_test(self):
        """Run the complete voice authenticity test with local models"""
        print("🎯 VOICE AUTHENTICITY TEST WITH LOCAL MODELS")
        print("=" * 60)
        print("Testing OpenAI fine-tuned models AND local LM Studio models")
        print("=" * 60)
        
        # Load real voice examples
        voice_examples = self.load_voice_examples()
        self.results["voice_examples"] = voice_examples
        
        # Combine all models
        all_models = {**OPENAI_MODELS}
        if self.lm_studio_available:
            all_models.update({f"{k}_LOCAL": v for k, v in LOCAL_MODELS.items()})
        
        print(f"📊 Testing {len(all_models)} total models:")
        print(f"   • {len(OPENAI_MODELS)} OpenAI models")
        print(f"   • {len(LOCAL_MODELS) if self.lm_studio_available else 0} Local models")
        
        # Test each scenario
        for scenario in VOICE_TEST_SCENARIOS:
            scenario_id = f"scenario_{scenario['id']}"
            print(f"\n🧪 Testing: {scenario['category']}")
            print(f"📝 Scenario: {scenario['scenario']}")
            print("-" * 60)
            
            self.results["test_responses"][scenario_id] = {
                "category": scenario['category'],
                "scenario": scenario['scenario'],
                "formal_version": scenario['formal_version'],
                "expected_elements": scenario['expected_voice_elements'],
                "responses": {}
            }
            
            model_count = 1
            # Test OpenAI models
            for model_name, model_id in OPENAI_MODELS.items():
                print(f"   {model_count}/{len(all_models)} 🤖 {model_name}...", end=" ")
                
                response = self.query_openai_model(model_id, scenario['scenario'])
                
                self.results["test_responses"][scenario_id]["responses"][model_name] = {
                    "model_id": model_id,
                    "model_type": "openai",
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                if "ERROR:" in response:
                    print("❌")
                else:
                    print("✅")
                
                model_count += 1
                time.sleep(0.5)  # Rate limiting
            
            # Test Local models
            if self.lm_studio_available:
                for model_name, model_id in LOCAL_MODELS.items():
                    display_name = f"{model_name}_LOCAL"
                    print(f"   {model_count}/{len(all_models)} 🖥️  {display_name}...", end=" ")
                    
                    response = self.query_local_model(model_id, scenario['scenario'])
                    
                    self.results["test_responses"][scenario_id]["responses"][display_name] = {
                        "model_id": model_id,
                        "model_type": "local",
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if "ERROR:" in response:
                        print("❌")
                    else:
                        print("✅")
                    
                    model_count += 1
                    time.sleep(1)  # Longer pause for local models
            
            print(f"✅ {scenario['category']} complete!")
        
        # Save results
        results_file = f"voice_test_with_local_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Test results saved to: {results_file}")
        
        return self.results
    
    def evaluate_voice_authenticity(self, results):
        """Use Claude Opus-4 to evaluate voice authenticity including local models"""
        print(f"\n🧠 VOICE AUTHENTICITY EVALUATION WITH CLAUDE OPUS-4")
        print("=" * 60)
        print("Evaluating ALL models (OpenAI + Local) for voice authenticity")
        
        evaluations = {}
        
        # Load voice examples for context
        voice_examples = results.get("voice_examples", [])
        voice_context = "\n".join([f"- {ex}" for ex in voice_examples[:5]])
        
        for scenario_id, scenario_data in results["test_responses"].items():
            print(f"   📊 Evaluating {scenario_data['category']}...")
            
            # Create evaluation prompt
            eval_prompt = f"""You are evaluating whether AI models captured Arjun's authentic writing voice.

ARJUN'S AUTHENTIC VOICE CHARACTERISTICS (from training data):
{voice_context}

KEY VOICE ELEMENTS TO LOOK FOR:
- Casual, direct communication with contractions
- Conversational tone, not formal
- Short, punchy sentences  
- Questions and practical focus
- Informal terms when appropriate
- Personal touch, not corporate speak

SCENARIO: {scenario_data['category']}
TASK: {scenario_data['scenario']}
FORMAL VERSION: {scenario_data['formal_version']}

MODEL RESPONSES (including local models):
"""
            
            for model_name, response_data in scenario_data["responses"].items():
                if not response_data['response'].startswith("ERROR:"):
                    model_type = "🖥️ LOCAL" if "LOCAL" in model_name else "☁️ OPENAI"
                    eval_prompt += f"\n--- {model_name} ({model_type}) ---\n{response_data['response']}\n"
            
            eval_prompt += """
EVALUATION REQUIRED:

1. **Voice Authenticity Score (1-10)** for each response:
   - How much does this sound like Arjun's actual voice?
   - Consider: contractions, casual tone, directness, personal style

2. **Specific Analysis** for each response:
   - What voice elements are present/missing?
   - Does it sound natural or artificially formal?
   - Would you believe Arjun actually wrote this?

3. **Ranking**: Which model best captured Arjun's authentic voice? Include both OpenAI and local models.

4. **Local vs OpenAI Comparison**: How do the local fine-tuned models compare to OpenAI fine-tuned models?

Be specific about voice characteristics and compare local models directly to OpenAI models."""
            
            try:
                evaluation = self.anthropic_client.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": eval_prompt}]
                )
                
                evaluations[scenario_id] = {
                    "category": scenario_data['category'],
                    "evaluation": evaluation.content[0].text,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"✅ {scenario_data['category']} evaluated")
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error evaluating {scenario_id}: {e}")
                evaluations[scenario_id] = {
                    "category": scenario_data['category'],
                    "error": str(e)
                }
        
        # Save evaluation results
        eval_file = f"voice_evaluation_with_local_{self.timestamp}.json"
        final_results = {
            **results,
            "voice_evaluations": evaluations,
            "completed_at": datetime.now().isoformat()
        }
        
        with open(eval_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"✅ Voice evaluations saved to: {eval_file}")
        return final_results

def main():
    """
Main entry point for the voice authenticity testing program.

Orchestrates the complete testing workflow:
1. Initializes the tester with API clients
2. Runs scenario-based tests across all models
3. Evaluates results using Claude Opus-4
4. Saves comprehensive results to JSON files
5. Displays preview of evaluation results
"""
    print("🎯 ARJUN VOICE AUTHENTICITY TEST WITH LOCAL MODELS")
    print("=" * 70)
    print("Purpose: Test OpenAI fine-tuned AND local LM Studio models")
    print("Method: Scenarios matching your real communication patterns")
    print("Evaluation: Claude Opus-4 rates all models for voice authenticity")
    print("=" * 70)
    
    tester = LocalModelTester()
    
    # Run voice tests
    results = tester.run_voice_test()
    
    # Get Claude evaluation
    final_results = tester.evaluate_voice_authenticity(results)
    
    # Show quick preview
    print(f"\n📊 QUICK PREVIEW")
    print("=" * 40)
    for scenario_id, eval_data in final_results.get("voice_evaluations", {}).items():
        if "evaluation" in eval_data:
            print(f"\n{eval_data['category']}:")
            eval_lines = eval_data['evaluation'].split('\n')[:3]
            for line in eval_lines:
                if line.strip():
                    print(f"  {line.strip()}")
    
    print(f"\n🎉 COMPREHENSIVE VOICE AUTHENTICITY TEST COMPLETE!")
    print(f"📁 Results include both OpenAI and local model comparisons!")

if __name__ == "__main__":
    main()