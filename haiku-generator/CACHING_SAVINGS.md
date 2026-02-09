# Prompt Caching Savings Analysis

## Overview

Prompt caching allows you to reuse system prompts across multiple API calls, paying only 12.5% of the normal cost for cached reads.

## Cost Structure

| Token Type | Cost per MTok | Multiplier |
|-----------|---------------|------------|
| Normal Input | $0.80 | 1.00x |
| Cache Creation | $1.00 | 1.25x |
| Cache Read | $0.10 | 0.125x |
| Output | $4.00 | 5.00x |

**Cache Duration**: 5 minutes

## Where Caching Helps

### Without Caching (quality_filter.py)

Every evaluation call sends the full system prompt:

```
Call 1: 1,500 token system prompt × $0.80 = $0.0012
Call 2: 1,500 token system prompt × $0.80 = $0.0012
Call 3: 1,500 token system prompt × $0.80 = $0.0012
...
Call 25: 1,500 token system prompt × $0.80 = $0.0012

Total for 25 calls: $0.0300
```

### With Caching (quality_filter_cached.py)

```
Call 1: 1,500 tokens (cache creation) × $1.00 = $0.0015
Call 2: 1,500 tokens (cache read) × $0.10 = $0.0002 ← 87.5% savings!
Call 3: 1,500 tokens (cache read) × $0.10 = $0.0002
...
Call 25: 1,500 tokens (cache read) × $0.10 = $0.0002

Total for 25 calls: $0.0063
```

**Savings**: $0.0237 (79% reduction on system prompts)

## Full Process Breakdown

### 2500 → 250 Haiku Generation

#### Phase 1: Generation (10 batches)
- **Cost**: ~$1.20
- **Caching**: Not applicable (different prompts each time)

#### Phase 2: Computational Filtering
- **Cost**: $0 (local processing)

#### Phase 3: First Evaluation (20 batches of 100)
- **Without caching**: ~$0.45
  - System prompt: 1,500 tokens × 20 calls = 30,000 tokens × $0.80 = $0.024
  - User prompts: ~15,000 tokens × 20 calls = 300,000 tokens × $0.80 = $0.24
  - Output: ~50,000 tokens × 20 calls × $4.00 = $0.20
  - **Subtotal**: ~$0.46

- **With caching**: ~$0.43
  - System prompt creation: 1,500 tokens × $1.00 = $0.0015 (first call)
  - System prompt reads: 1,500 tokens × 19 calls × $0.10 = $0.0029 (subsequent)
  - User prompts: same $0.24
  - Output: same $0.20
  - **Subtotal**: ~$0.43
  - **Savings**: $0.03

#### Phase 4: Final Evaluation (5 batches of 100)
- **Without caching**: ~$0.12
  - System prompt: 1,500 tokens × 5 calls × $0.80 = $0.006
  - User prompts: ~15,000 tokens × 5 calls × $0.80 = $0.06
  - Output: ~15,000 tokens × 5 calls × $4.00 = $0.06
  - **Subtotal**: ~$0.126

- **With caching**: ~$0.12
  - System prompt creation: $0.0015 (first call)
  - System prompt reads: 1,500 × 4 × $0.10 = $0.0006
  - User prompts: same $0.06
  - Output: same $0.06
  - **Subtotal**: ~$0.12
  - **Savings**: $0.006

## Total Savings Per Run

| Metric | Without Caching | With Caching | Savings |
|--------|----------------|--------------|---------|
| Generation | $1.20 | $1.20 | $0 |
| First Eval | $0.46 | $0.43 | $0.03 |
| Final Eval | $0.13 | $0.12 | $0.01 |
| **TOTAL** | **$1.79** | **$1.75** | **$0.04** |

**Savings per run**: ~$0.04 (2.2%)

## Why Smaller Than Expected?

The savings are modest because:
1. **System prompts are small** (~1,500 tokens) compared to user prompts (~15,000 tokens)
2. **Output costs dominate** ($4/MTok vs $0.80/MTok)
3. **Most tokens are unique** (the haiku being evaluated)

## When Caching Helps More

Caching provides bigger savings when:
- System prompts are **large** (10,000+ tokens)
- Many calls reuse the **same instructions**
- Output is **small** relative to input

## Real-World Example Output

```
PHASE 3: FIRST EVALUATION PASS (CACHED)
----------------------------------------------------------------------
First pass evaluation (1993 haiku → top 500)...
  Batch 1: $0.0234 (created cache: 1,847 tokens)
  Batch 2: $0.0212 (cache read: 1,847 tokens, saved $0.0015)
  Batch 3: $0.0212 (cache read: 1,847 tokens, saved $0.0015)
  ...
  Cache savings: $0.0285
```

## Recommendation

**Use caching** if:
- ✅ You run the script multiple times per day
- ✅ You want every bit of savings
- ✅ You're okay with slightly more complex code

**Skip caching** if:
- ❌ You only run once per day
- ❌ Simplicity is more important than 2% savings
- ❌ You're modifying prompts frequently (breaks cache)

## How to Use

Replace `quality_filter.py` with `quality_filter_cached.py`:

```bash
# Without caching (~$1.79)
python quality_filter.py

# With caching (~$1.75)
python quality_filter_cached.py
```

The cached version automatically:
1. Creates cache on first batch
2. Reuses cache for remaining batches (5 min window)
3. Shows cache savings in output
4. Creates TWO editions:
   - Curated Edition: Top 250 haiku with quality scores
   - Extended Edition: All filtered haiku (~1,900-2,000)
