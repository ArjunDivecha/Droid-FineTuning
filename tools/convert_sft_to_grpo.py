#!/usr/bin/env python3
"""
Convert SFT-style JSONL to GRPO JSONL (prompt/answer/system) and split train/valid.

Accepts the following input formats per line:
- {"messages": [{"role":"system"|"user"|"assistant","content":"..."}, ...]}
- {"instruction": "...", "response": "...", "system": "..."}

Usage:
  python3 tools/convert_sft_to_grpo.py \
    --input "/path/to/sft.jsonl" \
    --out-dir "/path/to/gspo_voice" \
    --valid-ratio 0.1 \
    --system "You are Arjun..." 
"""
import argparse, json, random
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--valid-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--system', type=str, default=(
        "You are Arjun Divecha. Write with a data-driven, nuanced investment voice. "
        "Be concise, structured, and include context, examples, and caveats."
    ))
    return ap.parse_args()

def extract_grpo(obj, default_system):
    # messages format
    if 'messages' in obj and isinstance(obj['messages'], list):
        sys = default_system
        prompt = ''
        answer = ''
        for m in obj['messages']:
            role = m.get('role')
            cnt = m.get('content','')
            if role == 'system' and cnt:
                sys = cnt
            elif role == 'user' and cnt:
                prompt = cnt
            elif role == 'assistant' and cnt:
                answer = cnt
        if prompt and answer:
            return {"prompt": prompt, "answer": answer, "system": sys}
        return None
    # instruction/response
    if 'instruction' in obj and 'response' in obj:
        return {
            "prompt": str(obj.get('instruction','')), 
            "answer": str(obj.get('response','')),
            "system": str(obj.get('system') or default_system)
        }
    # alpaca-like
    if 'prompt' in obj and 'answer' in obj:
        return {
            "prompt": str(obj.get('prompt','')),
            "answer": str(obj.get('answer','')),
            "system": str(obj.get('system') or default_system)
        }
    return None

def main():
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    with inp.open('r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ex = extract_grpo(obj, args.system)
            if ex and ex['prompt'].strip() and ex['answer'].strip():
                data.append(ex)

    random.Random(args.seed).shuffle(data)
    n = len(data)
    n_valid = max(1, int(n * args.valid_ratio))
    valid = data[:n_valid]
    train = data[n_valid:]

    with (out_dir / 'train.jsonl').open('w', encoding='utf-8') as ft:
        for ex in train:
            ft.write(json.dumps(ex, ensure_ascii=False) + '\n')
    with (out_dir / 'valid.jsonl').open('w', encoding='utf-8') as fv:
        for ex in valid:
            fv.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"Converted {n} examples → train={len(train)}, valid={len(valid)} → {out_dir}")

if __name__ == '__main__':
    main()

