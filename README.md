# cc_compaction

Demonstrates and tests the four context management techniques used internally by Claude Code, reproduced using the plain Anthropic Python API. All tests use real API calls against `claude-sonnet-4-6`.

## Techniques

| # | Technique | How |
|---|-----------|-----|
| 1 | **Full Compaction** | Send history + summarization prompt → replace history with summary |
| 2 | **Time-Based Micro Compaction** | Client-side: if gap since last assistant message > 60 min (cache TTL), replace old tool result bodies with `[Old tool result content cleared]` |
| 3 | **API Thinking Clearing** | `context-management-2025-06-27` beta + `clear_thinking_20251015` directive |
| 4 | **API Tool Clearing** | `context-management-2025-06-27` beta + `clear_tool_uses_20250919` directive |

## Results

> **Note:** Token savings are measured only for **API Tool Clearing** and **API Thinking Clearing** (techniques 3 & 4). These are the only techniques that reduce `input_tokens` reported by the API and are therefore directly measurable in a before/after comparison.
>
> **Full Compaction** and **Time-Based Micro Compaction** are tested for correctness only (right messages replaced, facts preserved) — not token savings. Full compaction resets context entirely so a simple before/after token count is not meaningful; time-based MC is pure client-side string replacement with no API call involved.

### API Tool Clearing — 10-turn agent loop (2000-char tool results per turn)

```
Turn:      1     2     3     4     5     6     7     8     9    10
Baseline: 1141  1712  2283  2854  3425  3996  4567  5138  5709  6280
Cleared:  1141  1712  2283  2357  2431  2505  2579  2653  2727  2801
```

| | Cumulative tokens | Final turn tokens |
|---|---|---|
| Baseline | 37,105 | 6,280 |
| Tool clearing | 23,189 | 2,801 |
| **Saving** | **37.5%** | **55.4%** |

---

### API Thinking Clearing — 8-turn agent loop (budget 3000 tokens/turn)

```
Turn:      1    2    3    4     5     6     7     8
Baseline: 640  763  984  1307  1519  1747  1992  2220
Cleared:  640  763  890  1143  1292  1249  1392  1534
```

| | Cumulative tokens | Final turn tokens |
|---|---|---|
| Baseline | 11,172 | 2,220 |
| Thinking clearing | 8,903 | 1,534 |
| **Saving** | **20.3%** | **30.9%** |

---

### API Tool Clearing + API Thinking Clearing Combined — 8-turn agent loop

```
Turn:      1     2     3     4     5     6     7     8
Baseline: 640  1763  1985  4346  4511  6753  6930  9166
Tool only: 640  1763  2058  4329  4519  4758  4967  5206
Think only:640  1763  1912  4153  4292  6535  6705  8929
Combined:  640  1763  1851  4092  4184  4428  4557  4795
```

| Strategy | Cumulative | Saving |
|----------|-----------|--------|
| Baseline | 36,094 | — |
| Tool clearing only | 28,240 | 21.8% |
| Thinking clearing only | 34,929 | 3.2% |
| **Combined** | **26,310** | **27.1%** |

## Key Findings

1. **Savings are roughly additive** — combined (27.1%) ≈ tool-only (21.8%) + thinking-only (3.2%), with a small positive interaction effect.
2. **Tool clearing dominates** when tool results are large. Thinking blocks contribute a smaller share unless `budget_tokens` is very high.
3. **`clear_thinking_20251015` must be first** in the `edits` array when combining both directives — the API returns 400 otherwise.
4. **`tool_choice: "any"` is incompatible with thinking mode** — the API rejects it. Drive tool use via prompt instead.
5. **Time-based MC is purely client-side** — no special API features needed. The 60-minute threshold matches the Anthropic prompt cache TTL.

## Setup

```bash
cp .env.example .env
# add your ANTHROPIC_API_KEY

uv run pytest test_compaction.py -v -s
```

Requires Python 3.11+. Dependencies installed automatically by `uv`.

## Test Coverage

| Test | What it proves |
|------|----------------|
| `test_full_compaction_produces_summary` | Summary generated from conversation |
| `test_compacted_context_answers_followup` | Summary preserves facts for follow-up |
| `test_time_based_mc_clears_old_results` | Old tool results replaced with marker |
| `test_time_based_mc_keeps_recent` | Most recent N results untouched |
| `test_time_based_mc_does_not_trigger_when_recent` | No clearing when gap < threshold |
| `test_api_thinking_clearing_reduces_tokens` | Fewer `input_tokens` with thinking clearing |
| `test_api_tool_clearing_reduces_tokens` | Fewer `input_tokens` with tool clearing |
| `test_agent_loop_cumulative_tokens_lower_with_clearing` | 37.5% cumulative savings, 10-turn tool loop |
| `test_agent_loop_late_turn_tokens_lower_with_clearing` | 55% fewer tokens on final turn |
| `test_thinking_loop_cumulative_tokens_lower_with_clearing` | 20.3% cumulative savings, 8-turn thinking loop |
| `test_thinking_loop_late_turn_tokens_lower_with_clearing` | 31% fewer tokens on final thinking turn |
| `test_combined_clearing_beats_individual` | Combined beats either strategy alone (27.1% saving) |
