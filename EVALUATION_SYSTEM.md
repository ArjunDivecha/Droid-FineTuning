# Adapter Evaluation System

## Overview

The evaluation system measures how well trained adapters maintain faithfulness to their training data while improving performance. It evaluates adapters on four key metrics:
- **Faithfulness**: How well the response matches training data
- **Fact Recall**: Whether key facts from training are included
- **Consistency**: Absence of contradictions with training data
- **Hallucination**: Addition of information not in training data

## Architecture

```
backend/
└── evaluation_api.py          # FastAPI endpoints for evaluation
adapter_fusion/
└── evaluate_adapters.py       # Core evaluation logic
```

## How It Works

1. **Load Training Data**: Extract Q&A pairs from training data
2. **Select Test Questions**: Randomly sample questions for evaluation
3. **Generate Responses**: Use MLX to generate responses from adapter
4. **Evaluate Quality**: Use Cerebras to score responses against training
5. **Aggregate Scores**: Calculate overall and category scores

## API Endpoints

### Start Evaluation
```
POST /api/evaluation/start
```

**Request Body:**
```json
{
  "adapter_name": "string",           // Required: Name of adapter to evaluate
  "training_data_path": "string",     // Optional: Path to training data
  "num_questions": "integer",         // Optional: Number of questions (default: 20)
  "evaluate_base_model": "boolean"    // Optional: Evaluate base model instead (default: false)
}
```

**Response:**
```json
{
  "success": true,
  "message": "Evaluation started",
  "adapter_name": "adapter_name"
}
```

### Get Evaluation Status
```
GET /api/evaluation/status
```

**Response:**
```json
{
  "running": true,
  "progress": 45,
  "current_question": 9,
  "total_questions": 20,
  "adapter_name": "adapter_name",
  "error": null
}
```

### Get Evaluation Result
```
GET /api/evaluation/result
```

**Response:**
```json
{
  "success": true,
  "result": {
    "adapter_name": "string",
    "is_base_model": "boolean",
    "adapter_config": "object",
    "training_data_path": "string",
    "num_questions": "integer",
    "evaluation_date": "string",
    "scores": {
      "overall": "number",
      "faithfulness": "number",
      "fact_recall": "number",
      "consistency": "number",
      "hallucination": "number"
    },
    "detailed_results": "array"
  }
}
```

## Evaluation Process

### 1. Training Data Loading
The system supports multiple data formats:
- **GRPO format**: `{"prompt": "...", "answer": "..."}`
- **Standard format**: `{"question": "...", "answer": "..."}`
- **Chat format**: `{"messages": [...]}` with user/assistant roles
- **Text format**: Q&A pairs in `"text"` field

### 2. Question Selection
Randomly samples questions from training data:
- Default: 20 questions
- Can specify custom number
- Handles datasets smaller than requested sample size

### 3. Response Generation
Uses MLX to generate responses:
- Loads base model and adapter
- Generates response for each question
- Handles timeouts and errors gracefully

### 4. Quality Evaluation
Uses Cerebras LLM to evaluate responses:
- Compares model response to training data answer
- Scores on 4 metrics (1-10 scale, converted to 1-100)
- Provides detailed analysis of facts recalled/missing/added

### 5. Result Aggregation
Calculates weighted average scores:
- Overall score = average of all 4 metrics
- Detailed results per question
- Exportable JSON and text reports

## Configuration

### Environment Variables
Create a `.env` file in `adapter_fusion/` directory:
```
CEREBRAS_API_KEY=your_api_key_here
```

### Training Data Path Resolution
If not explicitly provided, the system tries to find training data:
1. Session files in `local_qwen/sessions/`
2. Adapter configuration file
3. JSONL files in training data directory

## Best Practices

1. **Representative Sampling**: Use 10-30 questions for meaningful evaluation
2. **Include Validation Data**: If available, use separate validation set
3. **Monitor Progress**: Check status endpoint during evaluation
4. **Compare Results**: Evaluate base model alongside adapter for comparison
5. **Review Details**: Check detailed results for specific failure patterns

## Troubleshooting

### "Training data path not found"
- Verify training data exists at specified path
- Check that adapter name matches session/config files
- Ensure proper file permissions

### "Evaluation failed" or timeout errors
- Check Cerebras API key in .env file
- Verify MLX model paths in adapter config
- Reduce number of questions for faster evaluation

### Inaccurate scores
- Ensure training data format is consistent
- Check that questions have clear answers
- Verify adapter has finished training
