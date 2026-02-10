# Training Process Monitoring Fixes - Summary

## Date: 2026-02-09
## Audited by: Senior Developer

---

## Critical Issues Found and Fixed

### 1. **CRITICAL: Regex Pattern Failed on Negative Loss Values**

**Problem:** The original regex patterns couldn't match negative loss values like `-0.001` or `-1.234`.

**Original Code:**
```python
loss_pattern = re.compile(r'(?:^|(?<!Val ))(?:Train )?loss ([0-9.]+)', re.IGNORECASE)
val_pattern = re.compile(r'Val loss ([0-9.]+)')
```

**Fixed Code:**
```python
loss_pattern = re.compile(
    r'(?:^|(?<!Val\s))(?:Train\s+)?[Ll]oss[:\s]+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)',
    re.IGNORECASE
)
val_pattern = re.compile(r'Val\s+loss\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)', re.IGNORECASE)
```

**Changes:**
- Added `-?` to match optional negative sign
- Added support for scientific notation (`e-4`, `E+3`, etc.)
- Made whitespace matching more robust (`\s+`)
- Added support for colon format (`Loss: 0.123`)

---

### 2. **MAJOR: Monitoring Loop Could Block Indefinitely**

**Problem:** The original code used blocking `readline()` which could hang if the process stopped producing output.

**Original Code:**
```python
while self.current_process and self.current_process.poll() is None:
    output = self.current_process.stdout.readline()
    if not output:
        await asyncio.sleep(0.1)
        continue
```

**Fixed Code:**
```python
import select

while self.current_process and self.current_process.poll() is None:
    # Use select to check if there's data available with timeout
    stdout_fd = self.current_process.stdout.fileno()
    readable, _, _ = select.select([stdout_fd], [], [], 0.1)
    
    if not readable:
        consecutive_empty_reads += 1
        if consecutive_empty_reads >= max_empty_reads:
            logger.warning("No output from training process...")
        await asyncio.sleep(0.1)
        continue
    
    output = self.current_process.stdout.readline()
    consecutive_empty_reads = 0
```

**Changes:**
- Added `select.select()` with timeout to prevent blocking
- Added counter for consecutive empty reads
- Added health check logging when no output detected

---

### 3. **MEDIUM: Subprocess Output Buffering**

**Problem:** Python's output buffering could delay or lose training log messages.

**Fixed Code:**
```python
# Set environment variable to force Python to use unbuffered output
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

self.current_process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,  # Line buffered
    universal_newlines=True,
    preexec_fn=os.setsid,
    env=env  # Pass modified environment
)
```

**Also Fixed in local_lora.py:**
Added `flush=True` to all print statements:
```python
print(f"Iter {it}: Train loss {train_loss:.3f}, ...", flush=True)
```

---

### 4. **MEDIUM: Final Output Reading Could Hang**

**Problem:** Reading remaining output after process completion could block indefinitely.

**Fixed Code:**
```python
# Read any remaining output after process completion (with timeout)
final_output_count = 0
try:
    import select
    stdout_fd = self.current_process.stdout.fileno()
    
    # Try to read remaining output for up to 5 seconds
    for _ in range(50):  # 50 * 0.1s = 5 seconds
        readable, _, _ = select.select([stdout_fd], [], [], 0.1)
        if not readable:
            break
        
        remaining_output = self.current_process.stdout.readline()
        if not remaining_output:
            break
        # ... process output
```

---

### 5. **MINOR: Early Stopping Detection Was String-Based**

**Problem:** Used simple string matching `"Early stop:" in output` which could miss variations.

**Fixed Code:**
```python
early_stop_pattern = re.compile(r'Early\s+stop', re.IGNORECASE)

if early_stop_pattern.search(output):
    early_stop_detected = True
```

---

### 6. **MINOR: Added Error Detection Patterns**

**New Code:**
```python
error_patterns = [
    re.compile(r'Traceback\s+\(most\s+recent\s+call\s+last\)', re.IGNORECASE),
    re.compile(r'RuntimeError', re.IGNORECASE),
    re.compile(r'OutOfMemoryError', re.IGNORECASE),
    re.compile(r'CUDA\s+out\s+of\s+memory', re.IGNORECASE),
    re.compile(r'Killed', re.IGNORECASE),
]
```

---

### 7. **MINOR: Added Proper Resource Cleanup**

**New Code:**
```python
finally:
    # Ensure stdout is closed to prevent resource leaks
    if self.current_process and self.current_process.stdout:
        try:
            self.current_process.stdout.close()
            logger.debug("Closed training process stdout pipe")
        except Exception as e:
            logger.warning(f"Error closing stdout pipe: {e}")
```

---

## Files Modified

1. **backend/main.py**
   - Fixed regex patterns for metrics extraction
   - Added `select` import for non-blocking I/O
   - Added `PYTHONUNBUFFERED` environment variable
   - Added timeout-based output reading
   - Added error detection patterns
   - Added proper cleanup in finally block
   - Added comprehensive logging

2. **local_lora.py**
   - Added `flush=True` to remaining print statements

---

## Test Results

All tests pass successfully:
- ✓ Regex pattern validation (18 test cases)
- ✓ Validation vs Training loss separation
- ✓ Simulated training output parsing
- ✓ Negative loss value handling
- ✓ Scientific notation handling

---

## Supported Output Formats

The monitoring now correctly parses:

### Training Loss
- `Iter 10: Train loss 2.345`
- `Train loss 0.456`
- `loss 0.123`
- `Loss: 0.789`
- `Iter 10: Train loss -1.234` (negative values)
- `Iter 10: Train loss 1.23e-4` (scientific notation)

### Validation Loss
- `Iter 10: Val loss 2.123`
- `Val loss 1.234`
- `Iter 10: Val loss -0.001` (negative values)
- `Iter 10: Val loss 2.5E-3` (scientific notation)

### Learning Rate
- `Learning Rate 1.000e-05`
- `Learning rate 5e-6`

### Step Number
- `Iter 10:`
- `Iter 100:`
- `2025-01-09 10:23:45 Iter 15:` (with timestamp prefix)

---

## Verification

Run the test to verify all fixes:
```bash
python3 test_monitoring_fix.py
```

Run syntax check:
```bash
python3 -m py_compile backend/main.py
python3 -m py_compile local_lora.py
```

---

## Summary

All identified monitoring issues have been systematically fixed:

1. ✓ Regex patterns now match negative loss values
2. ✓ Regex patterns support scientific notation
3. ✓ Monitoring loop uses non-blocking I/O with timeouts
4. ✓ Subprocess uses unbuffered output
5. ✓ Final output reading has timeout protection
6. ✓ Error detection patterns added
7. ✓ Proper resource cleanup implemented
8. ✓ Comprehensive logging added

The training monitoring should now work reliably with mlx-lm-lora v1.0.1+ output formats.
