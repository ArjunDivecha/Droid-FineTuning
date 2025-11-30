#!/usr/bin/env python3
"""
Fix dataset conversion issues and create working UltraFeedback dataset.
"""

import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("/Users/macbook2024/Library/CloudStorage/Dropbox/Droid-FineTuning/gspo_datasets")

def create_working_ultrafeedback():
    """Create a working UltraFeedback dataset with proper field mapping."""
    print("🔧 Creating working UltraFeedback dataset...")
    
    try:
        # Try the alternative UltraFeedback dataset
        dataset = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
        
        gspo_data = []
        count = 0
        
        for example in dataset:
            if count >= 1000:  # Limit to 1000 examples for testing
                break
                
            # Extract data from UltraFeedback format
            if 'instruction' in example and 'response' in example:
                prompt = example['instruction']
                # UltraFeedback often has multiple responses, pick the first one
                if isinstance(example['response'], list) and len(example['response']) > 0:
                    answer = example['response'][0].get('content', '') if isinstance(example['response'][0], dict) else str(example['response'][0])
                else:
                    answer = str(example['response'])
                    
                if prompt and answer:
                    gspo_data.append({
                        "prompt": prompt.strip(),
                        "answer": answer.strip(),
                        "system": "You are a helpful AI assistant that provides accurate and informative responses."
                    })
                    count += 1
                    
            if count % 100 == 0:
                print(f"   Processed {count} examples...")
        
        # Split into train/validation
        train_size = int(len(gspo_data) * 0.9)
        train_data = gspo_data[:train_size]
        val_data = gspo_data[train_size:]
        
        # Save datasets
        with open(OUTPUT_DIR / "ultrafeedback_train.jsonl", 'w', encoding='utf-8') as f:
            for example in train_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(OUTPUT_DIR / "ultrafeedback_valid.jsonl", 'w', encoding='utf-8') as f:
            for example in val_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✅ Created UltraFeedback dataset: {len(train_data)} train, {len(val_data)} validation examples")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create UltraFeedback: {e}")
        return False

def create_simple_instruction_dataset():
    """Create a simple instruction dataset for testing."""
    print("🔧 Creating simple instruction dataset...")
    
    # Sample instruction-response pairs for testing
    instructions = [
        {
            "prompt": "Explain the concept of machine learning in simple terms.",
            "answer": "Machine learning is a way of teaching computers to learn patterns from data without being explicitly programmed. Think of it like teaching a child to recognize cats - you show them many pictures of cats, and eventually they learn to identify cats on their own. Similarly, machine learning algorithms analyze large amounts of data to recognize patterns and make predictions or decisions.",
            "system": "You are an expert educator who explains complex concepts in simple, understandable terms."
        },
        {
            "prompt": "What are the main differences between Python and JavaScript?",
            "answer": "Python and JavaScript are both popular programming languages, but they serve different purposes. Python is known for its simplicity and is widely used in data science, AI, and backend development. JavaScript is primarily used for web development, running in browsers to create interactive web pages. Python uses indentation for code structure while JavaScript uses curly braces. Python is statically typed with optional type hints, while JavaScript is dynamically typed. Python has a larger ecosystem for scientific computing, while JavaScript dominates web frontend development.",
            "system": "You are an experienced software engineer comparing programming languages."
        },
        {
            "prompt": "How does photosynthesis work?",
            "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in chloroplasts using chlorophyll. The process can be summarized in two main stages: the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, chlorophyll absorbs sunlight and uses water to produce oxygen and energy molecules (ATP and NADPH). In the Calvin cycle, plants use this energy to convert carbon dioxide into glucose (sugar). The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
            "system": "You are a biology teacher explaining natural processes clearly and accurately."
        },
        {
            "prompt": "What is the difference between weather and climate?",
            "answer": "Weather refers to the short-term atmospheric conditions in a specific place at a specific time, including temperature, humidity, precipitation, wind, and visibility. Climate, on the other hand, refers to the long-term patterns of weather in a region over extended periods, typically 30 years or more. A simple analogy: weather is what you wear today, while climate is what's in your wardrobe. Weather can change rapidly from day to day, while climate changes slowly over decades and centuries.",
            "system": "You are an environmental science educator explaining climate concepts."
        },
        {
            "prompt": "Explain the concept of compound interest.",
            "answer": "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Unlike simple interest, which is calculated only on the principal amount, compound interest allows your money to grow exponentially over time. For example, if you invest $1000 at 5% compound interest annually, after the first year you'll have $1050, after the second year you'll earn interest on $1050 (not just the original $1000), resulting in $1102.50. This snowball effect is why compound interest is often called the 'eighth wonder of the world' and is crucial for long-term investing.",
            "system": "You are a financial educator explaining investment concepts clearly."
        }
    ]
    
    # Split into train/validation
    train_data = instructions[:4]
    val_data = instructions[4:]
    
    # Save datasets
    with open(OUTPUT_DIR / "instruction_train.jsonl", 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open(OUTPUT_DIR / "instruction_valid.jsonl", 'w', encoding='utf-8') as f:
        for example in val_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created instruction dataset: {len(train_data)} train, {len(val_data)} validation examples")
    return True

def create_investopedia_dataset():
    """Create a working Investopedia-style financial dataset."""
    print("🔧 Creating financial dataset...")
    
    financial_examples = [
        {
            "prompt": "What is a P/E ratio and how is it used in stock valuation?",
            "answer": "The Price-to-Earnings (P/E) ratio is a valuation metric that compares a company's stock price to its earnings per share. It's calculated as: Stock Price ÷ Earnings Per Share. A P/E ratio of 20 means investors are willing to pay $20 for every $1 of current earnings. Generally, lower P/E ratios suggest better value, while higher P/E ratios may indicate growth expectations. However, P/E ratios should be compared within industries and to historical averages for meaningful analysis.",
            "system": "You are an expert financial analyst specializing in stock valuation and investment education."
        },
        {
            "prompt": "Explain the difference between stocks and bonds.",
            "answer": "Stocks represent ownership in a company, giving shareholders voting rights and potential for capital appreciation and dividends. Bonds are debt instruments where investors loan money to entities in exchange for periodic interest payments and return of principal at maturity. Stocks offer higher potential returns but with greater risk and volatility. Bonds provide more predictable income with lower risk. Stocks tend to perform well during economic expansions, while bonds often serve as safe havens during downturns. A balanced portfolio typically includes both for diversification.",
            "system": "You are an expert financial analyst explaining investment instruments."
        },
        {
            "prompt": "What is diversification and why is it important in investing?",
            "answer": "Diversification is the practice of spreading investments across various assets, sectors, industries, and geographic regions to reduce risk. The principle 'don't put all your eggs in one basket' applies directly to investing. By holding a variety of assets that don't move in perfect correlation, investors can reduce portfolio volatility without necessarily sacrificing returns. Diversification protects against the risk that any single investment performs poorly, as losses in one area may be offset by gains in another. Modern Portfolio Theory shows that diversification is the only 'free lunch' in investing, as it can lower risk while maintaining expected returns.",
            "system": "You are an expert portfolio manager explaining investment risk management."
        },
        {
            "prompt": "What is a mutual fund and how does it work?",
            "answer": "A mutual fund is a professionally managed investment vehicle that pools money from many investors to purchase a diversified portfolio of stocks, bonds, or other securities. Each investor owns shares representing their portion of the fund's holdings. Fund managers make investment decisions based on the fund's stated objectives. Mutual funds offer instant diversification, professional management, and liquidity (shares can be redeemed daily). They come in various types: equity funds (stocks), bond funds, balanced funds (mix), and money market funds. Investors pay expense ratios and may incur sales charges, but these costs provide access to professional management and diversification that might be difficult to achieve individually.",
            "system": "You are an expert financial educator explaining investment vehicles."
        },
        {
            "prompt": "Explain the concept of dollar-cost averaging.",
            "answer": "Dollar-cost averaging (DCA) is an investment strategy where an investor invests a fixed amount of money at regular intervals, regardless of market conditions. For example, investing $500 monthly in a mutual fund. This approach reduces the impact of market volatility by purchasing more shares when prices are low and fewer shares when prices are high. DCA eliminates the risk of making large investments at market peaks and removes emotional decision-making. While it may underperform lump-sum investing in consistently rising markets, it historically provides better risk-adjusted returns and helps investors avoid timing the market, which most professionals consider impossible to do consistently.",
            "system": "You are an expert financial advisor explaining investment strategies."
        }
    ]
    
    # Split into train/validation
    train_data = financial_examples[:4]
    val_data = financial_examples[4:]
    
    # Save datasets
    with open(OUTPUT_DIR / "finance_train.jsonl", 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open(OUTPUT_DIR / "finance_valid.jsonl", 'w', encoding='utf-8') as f:
        for example in val_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created financial dataset: {len(train_data)} train, {len(val_data)} validation examples")
    return True

def main():
    print("🔧 Fixing dataset conversion issues...")
    
    success_count = 0
    
    if create_working_ultrafeedback():
        success_count += 1
    
    if create_simple_instruction_dataset():
        success_count += 1
        
    if create_investopedia_dataset():
        success_count += 1
    
    print(f"\n✅ Successfully fixed {success_count}/3 datasets")
    
    # List final datasets
    print("\n📋 Final available datasets:")
    for file_path in sorted(OUTPUT_DIR.glob("*.jsonl")):
        if file_path.stat().st_size > 0:  # Only show non-empty files
            file_size = file_path.stat().st_size
            print(f"   📄 {file_path.name} ({file_size:,} bytes)")

if __name__ == "__main__":
    main()
