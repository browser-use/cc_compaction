# cc_compaction

Demonstrates and tests the four context management techniques used internally by Claude Code, reproduced using the plain Anthropic Python API.

## Techniques

**1. Full Compaction**
Send the conversation history with a summarization prompt. Replace the entire message history with the returned summary. Preserves key facts while resetting token count.

**2. Time-Based Micro Compaction**
Client-side logic: track the timestamp of the last assistant message. If the gap exceeds 60 minutes (the Anthropic prompt cache TTL), replace old tool result content with `[Old tool result content cleared]`, keeping the N most recent. Safe to do because the cache is already cold.

**3. API Thinking Clearing**
Send the `context-management-2025-06-27` beta header with a `clear_thinking_20251015` directive. The API drops old thinking blocks server-side before counting tokens, reducing cost on long reasoning sessions.

**4. API Tool Clearing**
Send the `context-management-2025-06-27` beta header with a `clear_tool_uses_20250919` directive. The API automatically drops old tool results above a token threshold. In a 10-turn agent loop with large tool results this produces ~37% cumulative token savings and ~55% fewer tokens on the final turn.

## Results (10-turn agent loop)

```
Turn:      1     2     3     4     5     6     7     8     9    10
Baseline: 1141  1712  2283  2854  3425  3996  4567  5138  5709  6280
Cleared:  1141  1712  2283  2357  2431  2505  2579  2653  2727  2801

Cumulative baseline: 37,105 tokens
Cumulative cleared:  23,189 tokens  (37.5% savings)
```

## Setup

```bash
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env

uv run pytest test_compaction.py -v -s
```

Requires Python 3.11+. Dependencies are managed via `pyproject.toml` and installed automatically by `uv`.

## Model

All tests run against `claude-sonnet-4-6`.

## Test Coverage

| Test | What it proves |
|------|---------------|
| `test_full_compaction_produces_summary` | Summary is generated from conversation |
| `test_compacted_context_answers_followup` | Summary preserves facts for follow-up |
| `test_time_based_mc_clears_old_results` | Old tool results are replaced with marker |
| `test_time_based_mc_keeps_recent` | Most recent N results are untouched |
| `test_time_based_mc_does_not_trigger_when_recent` | No clearing when gap < threshold |
| `test_api_thinking_clearing_reduces_tokens` | Fewer `input_tokens` with thinking clearing |
| `test_api_tool_clearing_reduces_tokens` | Fewer `input_tokens` with tool clearing |
| `test_agent_loop_cumulative_tokens_lower_with_clearing` | 37.5% cumulative savings over 10 turns |
| `test_agent_loop_late_turn_tokens_lower_with_clearing` | 55% fewer tokens on final turn |
