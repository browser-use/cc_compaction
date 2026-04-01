# Compaction Techniques — Build & Verify Plan

## Goal
Reproduce four context management techniques from Claude Code using the plain Anthropic Python API.

## Model
`claude-sonnet-4-6` used throughout.

## Techniques

### 1. Full Compaction
Send conversation history to Claude with a summarization prompt. Replace message history with the summary.

### 2. Time-Based Micro Compaction
Track timestamp of last assistant message. If gap > 60 minutes, replace old tool result content with `[Old tool result content cleared]` before sending, keeping the N most recent.

### 3. API Thinking Clearing
Send `context-management-2025-06-27` beta header with `clear_thinking_20251015` strategy to have the API automatically drop old thinking blocks.

### 4. API Tool Clearing
Send `context-management-2025-06-27` beta header with `clear_tool_uses_20250919` strategy to have the API automatically drop old tool results above a token threshold.

---

## Verification Criteria

| # | Test | Pass Condition |
|---|------|----------------|
| 1 | Full compaction produces a summary | Response contains a non-empty summary string covering the original conversation |
| 2 | Summary injected as new context works | Follow-up question answered correctly using only the summary (not original messages) |
| 3 | Time-based MC clears old tool results | Tool result content replaced with `[Old tool result content cleared]` for old results; recent ones intact |
| 4 | Time-based MC keeps N most recent | Exactly `keep_recent` tool results untouched |
| 5 | API thinking clearing accepted by API | Request with `clear_thinking_20251015` succeeds (no 400) |
| 6 | API tool clearing accepted by API | Request with `clear_tool_uses_20250919` succeeds (no 400) |
| 7 | API tool clearing triggers at threshold | Response usage shows reduced input tokens when threshold crossed |

---

## TDD Status

| Test | Status |
|------|--------|
| test_full_compaction_produces_summary | 🟢 GREEN |
| test_compacted_context_answers_followup | 🟢 GREEN |
| test_time_based_mc_clears_old_results | 🟢 GREEN |
| test_time_based_mc_keeps_recent | 🟢 GREEN |
| test_time_based_mc_does_not_trigger_when_recent | 🟢 GREEN |
| test_api_thinking_clearing_reduces_tokens | 🟢 GREEN |
| test_api_tool_clearing_reduces_tokens | 🟢 GREEN |
| test_agent_loop_cumulative_tokens_lower_with_clearing | 🟢 GREEN |
| test_agent_loop_late_turn_tokens_lower_with_clearing | 🟢 GREEN |
| test_thinking_loop_cumulative_tokens_lower_with_clearing | 🟢 GREEN |
| test_thinking_loop_late_turn_tokens_lower_with_clearing | 🟢 GREEN |
| test_combined_clearing_beats_individual | 🟢 GREEN |
