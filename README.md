# Anthropic Kernel Optimization Challenge — 1,358 Cycles

> **Why this matters for backend engineering:** This deep dive into low-level CPU instruction scheduling directly translates to building highly optimized, resource-efficient backend systems. The same principles of minimizing latency, maximizing throughput, and eliminating execution stalls are critical for scaling infrastructure on massive platforms like Salesforce.

Optimized a custom VLIW SIMD instruction scheduler to **1,358 cycles**, beating Claude Opus 4.5's best score of **1,363 cycles** on Anthropic's performance engineering take-home.

## The Challenge

Anthropic built a simulated custom processor — a VLIW (Very Large Instruction Word) SIMD machine — and the task is to write a kernel scheduler in Python that processes a random forest hash computation as fast as possible, measured in clock cycles on this simulated CPU.

The machine has:
- Multiple parallel execution engines per cycle: `alu` (×12 slots), `valu` (×6), `load` (×2), `store` (×2), `flow` (×1)
- SIMD vector width of 8 (`VLEN = 8`)
- 1536-word scratch space
- WAR (Write After Read) hazard constraints

The goal: fill instruction slots as densely as possible, exploit data-level parallelism, and eliminate pipeline stalls.

## My Score

| Optimizer | Cycles | Notes |
|-----------|--------|-------|
| Baseline (repo start) | ~18,532 | Naive sequential schedule |
| Claude Opus 4.5 (2hr harness) | 1,579 | Anthropic's AI benchmark |
| **Claude Opus 4.5 best (11.5hr harness)** | **1,363** | The target to beat |
| **My solution** | **1,358** | Beat the AI by 5 cycles |

Anthropic says: *"If you optimize below 1,487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com"*

![Leaderboard](Kernel%20Leaderboard.png)

## Key Optimizations

### 1. DAG-Based Instruction Scheduling
Built a full data-dependency graph (DAG) across all operations. Used a priority-queue topological sort (Kahn's algorithm) to schedule instructions in dependency order, maximizing the number of independent operations that can execute in the same cycle.

### 2. WAR Hazard Elimination
Enabled both `WAR_NODE` and `WAR_TMP` — inserts anti-dependency edges for Write-After-Read hazards. Prevents the scheduler from issuing a write before all prior reads of that address complete.

### 3. SIMD Vectorization (VALU)
Batched scalar operations into 8-wide VALU instructions wherever data is independent across the VLEN=8 lanes. This converts 8 sequential ALU slots into 1 VALU slot — the single biggest cycle reduction.

Core operations vectorized:
- XOR hash mixing (`val ^= node`)
- Bitwise masking (`idx & 1`) for branch-free tree traversal
- Multiply-add accumulation for hash finalization
- Bit shift + XOR for avalanche mixing

### 4. Group-Level Parallelism
Processes `batch_size // VLEN` groups in parallel. Each group holds a full 8-element SIMD batch. Allocating scratch vectors per group and interleaving their instructions lets the scheduler pack more work per cycle without data conflicts.

### 5. Slot Packing
The scheduler fills all available slots each cycle before advancing. With 12 ALU + 6 VALU + 2 load + 2 store slots per cycle, a well-ordered DAG keeps slot utilization near 100% in compute-heavy sections.

## Reproduce

```bash
# Verify solution (tests folder must be unmodified)
python tests/submission_tests.py

# Run the optimized kernel builder
python perf_takehome.py

# Visualize execution trace
python watch_trace.py    # then open watch_trace.html in browser
```

Expected output from `submission_tests.py`:
```
Cycle count: 1358
✓ Below 18532 (baseline)
✓ Below 4000
✓ Below 2164 (Claude Opus 4 many hours)
✓ Below 1790 (Claude Opus 4.5 casual)
✓ Below 1579 (Claude Opus 4.5 2hr harness)
✓ Below 1487 (Claude Opus 4.5 11.5hr harness)
✓ Below 1363 (Claude Opus 4.5 improved harness) ← beat the AI
```

## Files

| File | Purpose |
|------|---------|
| `perf_takehome.py` | **My solution** — `KernelBuilder.build_kernel()` |
| `problem.py` | Machine simulator (frozen, unmodified) |
| `tests/submission_tests.py` | Official benchmark (unmodified) |
| `watch_trace.py` / `watch_trace.html` | Cycle-level execution visualizer |
| `build_kernel_fast.py` | Alternate build variant |

## License

Per Anthropic's terms: permission granted to modify and use, but not to publish or redistribute solutions (to preserve the challenge for future candidates). This repo shares the approach and result without publishing a copy-pasteable solution.
