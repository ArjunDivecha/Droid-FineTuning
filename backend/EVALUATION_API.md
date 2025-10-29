# Evaluation API

## Overview

The Evaluation API provides endpoints for assessing the quality of trained adapters by measuring their faithfulness to training data. It evaluates adapters on key metrics including faithfulness, fact recall, consistency, and hallucination rates.

## Routes

### Start Evaluation
```
POST /api/evaluation/start
```

Starts evaluation of an adapter against its training data.

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

Retrieves the current status and progress of an evaluation.

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

Retrieves the results of a completed evaluation.

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

## Evaluation Metrics

### Faithfulness (1-100)
How well the model response matches the training data answer:
- 100 = Perfect match, all key facts from training present
- 50 = Partial match, some facts present
- 1 = No match, different information

### Fact Recall (1-100)
Whether the key facts from training data are included in the response:
- 100 = All key facts recalled
- 50 = Some facts recalled
- 1 = Few or no facts recalled

### Consistency (1-100)
Absence of contradictions with the training data:
- 100 = Fully consistent, no contradictions
- 50 = Minor inconsistencies
- 1 = Major contradictions

### Hallucination (1-100)
Addition of information not present in training data:
- 100 = Only uses trained information
- 50 = Some additions, mostly accurate
- 1 = Mostly made-up information

## Usage

1. **Start an evaluation:**
   ```bash
   curl -X POST http://localhost:8000/api/evaluation/start \
        -H "Content-Type: application/json" \
        -d '{"adapter_name": "my_adapter", "num_questions": 20}'
   ```

2. **Monitor progress:**
   ```bash
   curl http://localhost:8000/api/evaluation/status
   ```

3. **Get results:**
   ```bash
   curl http://localhost:8000/api/evaluation/result
   ```

## Requirements

- Cerebras API key in `.env` file
- Trained adapter in proper directory structure
- Training data in supported JSONL format
- MLX framework installed and configured

## Error Handling

- **400 Bad Request**: Evaluation already running
- **404 Not Found**: No evaluation result available
- **500 Internal Server Error**: Evaluation failed with details in response
