"""Single source of truth for the tool-calling prompt format.

Used by: build_llamafactory_data.py (SFT system field), grpo_to_verl.py
(RL prompts), evals/scripts/eval_transfer_matrix.py (eval prompts).
All three arms (SFT, GRPO, SFT->GRPO) and evaluation therefore see the
byte-identical system prompt and tool serialization — prompt format is a
controlled variable, not a confound (RESEARCH_PLAN.md §3.3).

Note: BFCL and the other external harnesses use each model's native tool
format; that is reported separately and identically for all arms.
"""
import json

SYSTEM_TMPL = (
    "You are a function-calling assistant. You may call one or more tools to "
    "answer the user. Available tools:\n{tools}\n"
    "For each tool call, output <tool_call>{{\"name\": <function-name>, "
    "\"arguments\": <args-dict>}}</tool_call>. "
    "If no available tool applies, answer directly without calling any tool."
)


def render_system(tools):
    return SYSTEM_TMPL.format(tools=json.dumps(tools, ensure_ascii=False))


def render_assistant_calls(calls):
    """Render tool calls the way the model is expected to emit them, so that
    multi-turn contexts show prior calls in the canonical output format."""
    return "\n".join(
        "<tool_call>" + json.dumps(c, ensure_ascii=False) + "</tool_call>"
        for c in calls
    )
