"""
Compaction techniques — TDD with real Anthropic API.
Run: uv run pytest test_compaction.py -v
"""

import os
import time
import pytest
import anthropic
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"
CLEARED_MARKER = "[Old tool result content cleared]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool_result_message(tool_use_id: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }

def make_tool_use_block(tool_use_id: str, name: str = "bash", input: dict | None = None) -> dict:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": input or {"command": "echo hi"}}


def build_conversation_with_tool_results(n: int) -> list[dict]:
    """Build a fake multi-turn conversation with n tool-use / tool-result pairs."""
    messages = [{"role": "user", "content": "Let's do some work."}]
    for i in range(n):
        tool_id = f"tool_{i:03d}"
        messages.append({
            "role": "assistant",
            "content": [make_tool_use_block(tool_id)],
        })
        messages.append(make_tool_result_message(tool_id, f"Output of command {i}: some long result data here."))
    return messages


# ---------------------------------------------------------------------------
# 1. Full Compaction
# ---------------------------------------------------------------------------

COMPACT_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
Your task is to create a concise summary of the conversation so far.
Include: primary intent, key facts established, current state.
Respond with a <summary>...</summary> block only."""


def compact_conversation(messages: list[dict]) -> str:
    """Send conversation + summarization prompt, return summary text."""
    summarize_request = {"role": "user", "content": COMPACT_PROMPT}
    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=messages + [summarize_request],
    )
    return response.content[0].text


def test_full_compaction_produces_summary():
    messages = [
        {"role": "user", "content": "My name is Ada and I am building a Python web scraper."},
        {"role": "assistant", "content": "Great! I can help you build a Python web scraper using requests and BeautifulSoup."},
        {"role": "user", "content": "I want to scrape hacker news headlines."},
        {"role": "assistant", "content": "For Hacker News, you can use the official API at hacker-news.firebaseio.com."},
    ]
    summary = compact_conversation(messages)
    assert len(summary) > 50
    assert "<summary>" in summary or "summary" in summary.lower() or "Ada" in summary or "scraper" in summary.lower()


def test_compacted_context_answers_followup():
    original_messages = [
        {"role": "user", "content": "The secret code word for this session is PINEAPPLE."},
        {"role": "assistant", "content": "Understood. The secret code word is PINEAPPLE."},
    ]
    summary = compact_conversation(original_messages)

    # Replace history with just the summary
    new_messages = [
        {
            "role": "user",
            "content": f"This session continues from a previous conversation.\n\n{summary}",
        },
        {"role": "assistant", "content": "Understood, I have the context from the previous conversation."},
        {"role": "user", "content": "What was the secret code word?"},
    ]
    response = CLIENT.messages.create(model=MODEL, max_tokens=256, messages=new_messages)
    answer = response.content[0].text
    assert "PINEAPPLE" in answer.upper()


# ---------------------------------------------------------------------------
# 2. Time-Based Micro Compaction
# ---------------------------------------------------------------------------

COMPACTABLE_TOOLS = {"bash", "file_read", "grep", "glob", "web_search", "web_fetch", "file_edit", "file_write"}
GAP_THRESHOLD_MINUTES = 60
KEEP_RECENT = 2


def time_based_mc(messages: list[dict], last_assistant_ts: float, keep_recent: int = KEEP_RECENT) -> list[dict]:
    """
    If gap since last assistant message > GAP_THRESHOLD_MINUTES, clear old
    compactable tool results, keeping the `keep_recent` most recent.
    """
    gap_minutes = (time.time() - last_assistant_ts) / 60
    if gap_minutes < GAP_THRESHOLD_MINUTES:
        return messages

    # Collect compactable tool_use ids in encounter order
    compactable_ids = []
    for msg in messages:
        if msg["role"] == "assistant":
            for block in (msg["content"] if isinstance(msg["content"], list) else []):
                if block.get("type") == "tool_use" and block.get("name") in COMPACTABLE_TOOLS:
                    compactable_ids.append(block["id"])

    keep_set = set(compactable_ids[-keep_recent:])
    clear_set = set(id_ for id_ in compactable_ids if id_ not in keep_set)

    result = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            new_content = []
            touched = False
            for block in msg["content"]:
                if (
                    block.get("type") == "tool_result"
                    and block.get("tool_use_id") in clear_set
                    and block.get("content") != CLEARED_MARKER
                ):
                    new_content.append({**block, "content": CLEARED_MARKER})
                    touched = True
                else:
                    new_content.append(block)
            result.append({**msg, "content": new_content} if touched else msg)
        else:
            result.append(msg)
    return result


def test_time_based_mc_clears_old_results():
    messages = build_conversation_with_tool_results(5)
    # Simulate last assistant message was 61 minutes ago
    old_ts = time.time() - 61 * 60

    result = time_based_mc(messages, last_assistant_ts=old_ts, keep_recent=2)

    # Collect all tool result contents
    results_content = []
    for msg in result:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    results_content.append(block["content"])

    cleared = [c for c in results_content if c == CLEARED_MARKER]
    intact = [c for c in results_content if c != CLEARED_MARKER]

    assert len(cleared) == 3  # 5 total - 2 kept = 3 cleared
    assert len(intact) == 2   # keep_recent=2


def test_time_based_mc_keeps_recent():
    messages = build_conversation_with_tool_results(4)
    old_ts = time.time() - 61 * 60

    result = time_based_mc(messages, last_assistant_ts=old_ts, keep_recent=2)

    tool_results = []
    for msg in result:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    tool_results.append(block)

    # Last 2 should be untouched
    assert tool_results[-1]["content"] != CLEARED_MARKER
    assert tool_results[-2]["content"] != CLEARED_MARKER
    # First 2 should be cleared
    assert tool_results[0]["content"] == CLEARED_MARKER
    assert tool_results[1]["content"] == CLEARED_MARKER


def test_time_based_mc_does_not_trigger_when_recent():
    messages = build_conversation_with_tool_results(5)
    recent_ts = time.time() - 5 * 60  # only 5 minutes ago

    result = time_based_mc(messages, last_assistant_ts=recent_ts, keep_recent=2)

    # Nothing should be cleared
    for msg in result:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    assert block["content"] != CLEARED_MARKER


# ---------------------------------------------------------------------------
# 3. API Thinking Clearing
# ---------------------------------------------------------------------------

THINKING_BETAS = ["context-management-2025-06-27", "interleaved-thinking-2025-05-14"]
THINKING_CONFIG = {"type": "enabled", "budget_tokens": 2000}


def build_thinking_history() -> list[dict]:
    """
    Make two real API calls with thinking enabled to produce a history
    with two genuine thinking blocks that the API will accept.
    """
    # Turn 1
    r1 = CLIENT.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=THINKING_BETAS,
        thinking=THINKING_CONFIG,
        messages=[{"role": "user", "content": "What is 3 * 7?"}],
    )
    turn1_assistant = {"role": "assistant", "content": [b.model_dump() for b in r1.content]}

    # Turn 2
    r2 = CLIENT.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=THINKING_BETAS,
        thinking=THINKING_CONFIG,
        messages=[
            {"role": "user", "content": "What is 3 * 7?"},
            turn1_assistant,
            {"role": "user", "content": "Now what is 6 * 8?"},
        ],
    )
    turn2_assistant = {"role": "assistant", "content": [b.model_dump() for b in r2.content]}

    history = [
        {"role": "user", "content": "What is 3 * 7?"},
        turn1_assistant,
        {"role": "user", "content": "Now what is 6 * 8?"},
        turn2_assistant,
        {"role": "user", "content": "What were the two answers?"},
    ]
    return history


def test_api_thinking_clearing_reduces_tokens():
    """
    clear_thinking_20251015 with keep=1 should consume fewer input tokens
    than keep='all' because old thinking blocks are dropped server-side.
    """
    history = build_thinking_history()

    # Baseline: keep all thinking blocks
    r_keep_all = CLIENT.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=THINKING_BETAS,
        thinking=THINKING_CONFIG,
        context_management={
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
        messages=history,
    )
    tokens_keep_all = r_keep_all.usage.input_tokens

    # With clearing: keep only 1 thinking turn — drops the first thinking block
    r_cleared = CLIENT.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=THINKING_BETAS,
        thinking=THINKING_CONFIG,
        context_management={
            "edits": [{"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 1}}]
        },
        messages=history,
    )
    tokens_cleared = r_cleared.usage.input_tokens

    assert tokens_cleared < tokens_keep_all, (
        f"Expected fewer tokens with thinking clearing: "
        f"keep_all={tokens_keep_all}, cleared={tokens_cleared}"
    )


# ---------------------------------------------------------------------------
# 4. API Tool Clearing
# ---------------------------------------------------------------------------

BASH_TOOL = {
    "name": "bash",
    "description": "Run a bash command",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}

# Large tool results so clearing produces a measurable token difference
LARGE_RESULT = "x" * 3000


def build_tool_result_history(n: int = 5) -> list[dict]:
    """n tool-use/result pairs followed by a final user question."""
    messages = []
    for i in range(n):
        tool_id = f"tool_{i:03d}"
        messages.append({"role": "user", "content": f"Run command {i}."})
        messages.append({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": "bash", "input": {"command": f"echo {i}"}}],
        })
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": f"Result {i}: {LARGE_RESULT}"}],
        })
    messages.append({"role": "user", "content": "Summarise what you did."})
    return messages


def test_api_tool_clearing_reduces_tokens():
    """
    clear_tool_uses_20250919 should reduce input_tokens compared to
    sending the same conversation without context_management.
    """
    messages = build_tool_result_history(n=5)

    # Baseline: no context management
    r_baseline = CLIENT.beta.messages.create(
        model=MODEL,
        max_tokens=256,
        betas=["context-management-2025-06-27"],
        tools=[BASH_TOOL],
        messages=messages,
    )
    tokens_baseline = r_baseline.usage.input_tokens

    # With tool clearing — threshold set very low to guarantee it fires
    r_cleared = CLIENT.beta.messages.create(
        model=MODEL,
        max_tokens=256,
        betas=["context-management-2025-06-27"],
        tools=[BASH_TOOL],
        context_management={
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": 100},
                    "clear_at_least": {"type": "input_tokens", "value": 50},
                }
            ]
        },
        messages=messages,
    )
    tokens_cleared = r_cleared.usage.input_tokens

    assert tokens_cleared < tokens_baseline, (
        f"Expected fewer tokens with tool clearing: "
        f"baseline={tokens_baseline}, cleared={tokens_cleared}"
    )


# ---------------------------------------------------------------------------
# 5. Long Agent Loop — cumulative & late-turn token comparison
# ---------------------------------------------------------------------------

LOOP_TURNS = 10
# Large tool result so clearing has a measurable effect per turn
LOOP_TOOL_RESULT = "y" * 2000
# Clearing fires as soon as history grows — guarantees it triggers every turn
LOOP_CLEAR_THRESHOLD = 200
LOOP_CLEAR_AT_LEAST = 100


def run_agent_loop(use_clearing: bool) -> list[int]:
    """
    Simulate a 10-turn agent loop by:
      1. Synthetically building up tool-use / tool-result history turn by turn.
      2. At each turn, making a real API call with the accumulated history and
         recording the reported input_tokens.

    Returns a list of input_token counts, one per turn.
    """
    context_mgmt = (
        {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": LOOP_CLEAR_THRESHOLD},
                    "clear_at_least": {"type": "input_tokens", "value": LOOP_CLEAR_AT_LEAST},
                }
            ]
        }
        if use_clearing
        else None
    )

    history: list[dict] = []
    token_counts: list[int] = []

    for i in range(LOOP_TURNS):
        tool_id = f"loop_tool_{i:03d}"

        # Append synthetic tool-use turn
        history.append({
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": "bash", "input": {"command": f"echo step_{i}"}}],
        })
        history.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": f"step_{i}: {LOOP_TOOL_RESULT}"}],
        })

        # Real API call — measure input_tokens at this turn
        kwargs: dict = dict(
            model=MODEL,
            max_tokens=64,
            betas=["context-management-2025-06-27"],
            tools=[BASH_TOOL],
            messages=[{"role": "user", "content": "Start counting steps."}] + history + [{"role": "user", "content": "How many steps so far?"}],
        )
        if context_mgmt:
            kwargs["context_management"] = context_mgmt

        response = CLIENT.beta.messages.create(**kwargs)
        token_counts.append(response.usage.input_tokens)

    return token_counts


def test_agent_loop_cumulative_tokens_lower_with_clearing():
    """Sum of input_tokens across all turns is lower with clearing enabled."""
    baseline_tokens = run_agent_loop(use_clearing=False)
    cleared_tokens = run_agent_loop(use_clearing=True)

    cumulative_baseline = sum(baseline_tokens)
    cumulative_cleared = sum(cleared_tokens)

    print(f"\nBaseline per-turn tokens: {baseline_tokens}")
    print(f"Cleared  per-turn tokens: {cleared_tokens}")
    print(f"Cumulative baseline: {cumulative_baseline}")
    print(f"Cumulative cleared:  {cumulative_cleared}")
    print(f"Savings: {cumulative_baseline - cumulative_cleared} tokens ({100 * (cumulative_baseline - cumulative_cleared) / cumulative_baseline:.1f}%)")

    assert cumulative_cleared < cumulative_baseline, (
        f"Expected clearing to reduce cumulative tokens: "
        f"baseline={cumulative_baseline}, cleared={cumulative_cleared}"
    )


def test_agent_loop_late_turn_tokens_lower_with_clearing():
    """
    On the final turn, input_tokens is lower with clearing — proving that
    history size is bounded rather than growing unboundedly.
    """
    baseline_tokens = run_agent_loop(use_clearing=False)
    cleared_tokens = run_agent_loop(use_clearing=True)

    last_baseline = baseline_tokens[-1]
    last_cleared = cleared_tokens[-1]

    print(f"\nTurn {LOOP_TURNS} baseline: {last_baseline}")
    print(f"Turn {LOOP_TURNS} cleared:  {last_cleared}")

    assert last_cleared < last_baseline, (
        f"Expected fewer tokens on final turn with clearing: "
        f"baseline={last_baseline}, cleared={last_cleared}"
    )
