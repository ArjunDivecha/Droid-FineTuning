#!/usr/bin/env python3.11
"""
Convert SFT dataset to GSPO format with intelligent chunking for VOICE/STYLE training.

Strategy:
- Chunk long articles into 400-700 word segments (paragraph-aware)
- Generate generic prompts focused on writing style, not content
- Preserve writing voice in answers
- Output GSPO format: {"prompt": "...", "answer": "...", "system": "..."}

Author: Claude Code
Date: 2025-10-08
"""

import json
import re
import os
from typing import List, Dict, Tuple
from pathlib import Path

# Configuration
MIN_CHUNK_WORDS = 300
MAX_CHUNK_WORDS = 700
TARGET_CHUNK_WORDS = 500

# System message emphasizing STYLE, not content expertise
STYLE_SYSTEM_MESSAGE = (
    "Write in a clear, analytical, and insightful investment commentary style. "
    "Use nuanced observations, data-driven insights, and thoughtful analysis. "
    "Blend narrative storytelling with factual precision."
)

# Generic prompt templates for style training (not content-specific)
PROMPT_TEMPLATES = [
    "Write an analytical piece about {theme}",
    "Discuss {theme} with data-driven insights",
    "Provide a thoughtful analysis of {theme}",
    "Explain {theme} with nuanced observations",
    "Analyze {theme} from an investment perspective",
]


def extract_theme_from_article(article: str, topic_hint: str = "") -> str:
    """
    Extract the main theme from article for generic prompt generation.
    Returns a generic theme, not specific content.
    """
    # Try to extract from topic hint first
    if topic_hint:
        # Clean up file paths and dates
        theme = topic_hint.split('/')[-1] if '/' in topic_hint else topic_hint
        # Remove dates like "20020828", "201107", "220301"
        theme = re.sub(r'\d{6,8}\s*', '', theme)
        # Remove "Emerging Thoughts -" prefix
        theme = re.sub(r'Emerging Thoughts\s*-\s*', '', theme, flags=re.IGNORECASE)
        # Clean up
        theme = theme.strip()
        if theme:
            return theme.lower()

    # Fallback: extract from article content
    # Look for title patterns or key themes
    lines = article.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        if line.strip() and len(line.strip()) > 10 and not line.startswith('"'):
            return line.strip().lower()

    return "market dynamics and investment strategies"


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (double newline separated)."""
    # Split on double newlines, preserving single newlines within paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    # Clean up each paragraph
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def split_long_paragraph(para: str, max_words: int) -> List[str]:
    """Split a long paragraph into sentence-based chunks."""
    # Split on sentence boundaries
    sentences = re.split(r'([.!?]+\s+)', para)
    # Rejoin sentence with its punctuation
    sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
    if len(sentences) < len(para.split('.')):
        sentences.append(sentences[-1])

    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words > max_words and current:
            chunks.append(' '.join(current))
            current = [sent]
            current_words = sent_words
        else:
            current.append(sent)
            current_words += sent_words

    if current:
        chunks.append(' '.join(current))

    return chunks


def chunk_article_by_paragraphs(article: str, min_words: int, max_words: int) -> List[str]:
    """
    Chunk article into segments of min_words to max_words, respecting paragraph boundaries.
    Force-split large paragraphs at sentence boundaries.
    """
    paragraphs = split_into_paragraphs(article)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para_word_count = len(para.split())

        # If single paragraph exceeds max, force split at sentence boundaries
        if para_word_count > max_words:
            # Finalize current chunk if exists
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_word_count = 0

            # Split this long paragraph
            para_chunks = split_long_paragraph(para, max_words)
            chunks.extend(para_chunks)
            continue

        # If adding this paragraph exceeds max, finalize current chunk
        if current_chunk and current_word_count + para_word_count > max_words:
            # Only finalize if we've met minimum
            if current_word_count >= min_words:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_word_count = para_word_count
            else:
                # Below minimum, add anyway and finalize
                current_chunk.append(para)
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_word_count = 0
        else:
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_word_count += para_word_count

    # Add remaining chunk if it meets minimum
    if current_chunk and current_word_count >= min_words:
        chunks.append('\n\n'.join(current_chunk))
    elif current_chunk and chunks:
        # Append to last chunk if too small
        chunks[-1] += '\n\n' + '\n\n'.join(current_chunk)
    elif current_chunk:
        # First chunk, keep even if small
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def generate_prompt(theme: str, chunk_index: int = 0) -> str:
    """Generate a generic style-focused prompt."""
    template = PROMPT_TEMPLATES[chunk_index % len(PROMPT_TEMPLATES)]
    return template.format(theme=theme)


def convert_sft_to_gspo_voice(
    input_file: str,
    output_dir: str,
    train_split: float = 0.85
) -> Tuple[int, int]:
    """
    Convert SFT dataset to GSPO format with voice-focused chunking.

    Returns: (num_train_examples, num_valid_examples)
    """
    # Read input
    with open(input_file, 'r') as f:
        lines = f.readlines()

    print(f"📖 Read {len(lines)} articles from {input_file}")

    # Process each article
    all_examples = []

    for line_idx, line in enumerate(lines):
        data = json.loads(line)

        # Extract messages
        system_msg = data['messages'][0]['content']
        user_msg = data['messages'][1]['content']
        assistant_msg = data['messages'][2]['content']

        # Extract topic hint from user message
        topic_hint = user_msg.replace('Write an article on:', '').strip()

        # Extract theme (generic)
        theme = extract_theme_from_article(assistant_msg, topic_hint)

        # Chunk the article
        chunks = chunk_article_by_paragraphs(
            assistant_msg,
            MIN_CHUNK_WORDS,
            MAX_CHUNK_WORDS
        )

        print(f"  Article {line_idx+1}: {len(assistant_msg.split())} words → {len(chunks)} chunks")

        # Create GSPO examples from chunks
        for chunk_idx, chunk in enumerate(chunks):
            prompt = generate_prompt(theme, chunk_idx)

            example = {
                "prompt": prompt,
                "answer": chunk,
                "system": STYLE_SYSTEM_MESSAGE
            }

            all_examples.append(example)

    print(f"\n✅ Created {len(all_examples)} training examples from {len(lines)} articles")
    print(f"   Average: {len(all_examples) / len(lines):.1f} chunks per article")

    # Calculate split
    num_train = int(len(all_examples) * train_split)
    train_examples = all_examples[:num_train]
    valid_examples = all_examples[num_train:]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write train split
    train_path = os.path.join(output_dir, 'train.jsonl')
    with open(train_path, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')

    # Write valid split
    valid_path = os.path.join(output_dir, 'valid.jsonl')
    with open(valid_path, 'w') as f:
        for example in valid_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n📊 Split:")
    print(f"   Train: {len(train_examples)} examples → {train_path}")
    print(f"   Valid: {len(valid_examples)} examples → {valid_path}")

    # Show sample
    print(f"\n📝 Sample training example:")
    sample = train_examples[0]
    print(f"   Prompt: {sample['prompt'][:80]}...")
    print(f"   Answer: {len(sample['answer'].split())} words")
    print(f"   System: {sample['system'][:80]}...")

    return len(train_examples), len(valid_examples)


def analyze_chunks(output_dir: str):
    """Analyze the generated chunks."""
    train_path = os.path.join(output_dir, 'train.jsonl')

    word_counts = []
    with open(train_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            word_count = len(data['answer'].split())
            word_counts.append(word_count)

    print(f"\n📊 Chunk Analysis:")
    print(f"   Min: {min(word_counts)} words")
    print(f"   Max: {max(word_counts)} words")
    print(f"   Avg: {sum(word_counts)//len(word_counts)} words")
    print(f"   Median: {sorted(word_counts)[len(word_counts)//2]} words")
    print(f"\n   Estimated tokens (×1.3):")
    print(f"   Avg: {int((sum(word_counts)//len(word_counts))*1.3)} tokens")
    print(f"   Max: {int(max(word_counts)*1.3)} tokens")


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/Writing/writing_dataset.jsonl"
    OUTPUT_DIR = "/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/gspo_voice"

    print("🔄 Converting SFT dataset to GSPO format (VOICE TRAINING)")
    print("=" * 70)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Chunk size: {MIN_CHUNK_WORDS}-{MAX_CHUNK_WORDS} words")
    print("=" * 70)
    print()

    # Convert
    num_train, num_valid = convert_sft_to_gspo_voice(
        INPUT_FILE,
        OUTPUT_DIR,
        train_split=0.85
    )

    # Analyze
    analyze_chunks(OUTPUT_DIR)

    print("\n✅ Conversion complete!")
    print(f"\n🚀 Ready for GSPO training in Enhanced Setup tab:")
    print(f"   - Select training method: GSPO")
    print(f"   - Training data: {OUTPUT_DIR}")
    print(f"   - Group size: 4-8 (more completions = better style learning)")
    print(f"   - Temperature: 0.8-1.0 (allow creative variation)")
    print(f"   - Iterations: {num_train * 10}-{num_train * 20} (10-20 epochs)")
