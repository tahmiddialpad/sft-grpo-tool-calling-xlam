#!/usr/bin/env python3
"""ToolRL-style verifiable reward for GRPO on tool calling (verl / EasyR1 compatible).

verl custom_reward_function entry point:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

ground_truth: JSON string of gold_calls, e.g. '[{"name": "f", "arguments": {"a": 1}}]'
(from datasets/grpo/*.jsonl). Reward in [0, 1]:
    0.2 * format + 0.3 * name-match + 0.5 * argument-match
Ablation flag BINARY_REWARD=1 switches to exact-match 0/1 (Tool-N1 style).
"""
import json, os, re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
# Qwen3.5's own SFT-learned tool-call syntax (LLaMA-Factory rendered its
# training targets this way for this architecture — differs from Qwen3's
# JSON <tool_call>{...}</tool_call> form, which the block above handles):
#   <function=NAME><parameter=KEY>VALUE</parameter>...</function>
XML_FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.S)
XML_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)
BINARY = os.environ.get("BINARY_REWARD", "0") == "1"


def _parse_xml_function_calls(text):
    out = []
    for name, body in XML_FUNCTION_RE.findall(text):
        args = {}
        for key, val in XML_PARAM_RE.findall(body):
            v = val.strip()
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            args[key.strip()] = v
        out.append({"name": name.strip(), "arguments": args})
    return out or None


def parse_calls(text):
    """Extract predicted tool calls from model output. Supports:
    - Qwen/Hermes: <tool_call>{...}</tool_call> blocks
    - bare JSON object or list: {"name":..., "arguments"|"parameters": {...}}
    - Qwen3.5 XML: <function=NAME><parameter=KEY>VALUE</parameter></function>
    Returns list of {"name", "arguments"} or None if unparseable."""
    xml_calls = _parse_xml_function_calls(text)
    if xml_calls:
        return xml_calls
    blocks = TOOL_CALL_RE.findall(text)
    if not blocks:
        # try fenced json
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        cand = [m.group(1)] if m else []
        # try first {...} or [...] span
        for opener, closer in (("[", "]"), ("{", "}")):
            s, e = text.find(opener), text.rfind(closer)
            if s != -1 and e > s:
                cand.append(text[s:e + 1])
        blocks = cand
    calls = []
    for b in blocks:
        try:
            obj = json.loads(b)
        except json.JSONDecodeError:
            continue
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if isinstance(it, dict) and "name" in it:
                args = it.get("arguments", it.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                calls.append({"name": it["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls or None


def _norm(v):
    if isinstance(v, str):
        s = v.strip().lower()
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                pass
        return s
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def match_call(pred, gold):
    """(name_ok, arg_score) for one pred/gold pair."""
    if pred["name"] != gold["name"]:
        return 0.0, 0.0
    ga, pa = gold.get("arguments", {}), pred.get("arguments", {})
    # Some source datasets (e.g. glaive) occasionally carry non-dict
    # `arguments` (observed: a list) — treat anything non-dict as unscorable
    # rather than crashing the whole rollout on one malformed gold record.
    if not isinstance(ga, dict):
        ga = {}
    if not isinstance(pa, dict):
        pa = {}
    if not ga:
        return 1.0, 1.0 if not pa else 0.5
    hit = sum(1 for k, v in ga.items() if k in pa and _norm(pa[k]) == _norm(v))
    extra = sum(1 for k in pa if k not in ga)
    score = hit / len(ga)
    if extra:
        score *= max(0.0, 1 - 0.25 * extra)
    return 1.0, score


def score_calls(pred_calls, gold_calls):
    """Greedy one-to-one matching; returns (name_frac, arg_frac, exact)."""
    if not gold_calls:  # irrelevance: correct behavior is NO call
        none_pred = not pred_calls
        return (1.0, 1.0, True) if none_pred else (0.0, 0.0, False)
    used, name_hits, arg_total = set(), 0, 0.0
    for g in gold_calls:
        best_i, best = None, (0.0, 0.0)
        for i, p in enumerate(pred_calls):
            if i in used:
                continue
            m = match_call(p, g)
            if m > best:
                best_i, best = i, m
        if best_i is not None and best[0] > 0:
            used.add(best_i)
            name_hits += 1
            arg_total += best[1]
    n = len(gold_calls)
    name_frac, arg_frac = name_hits / n, arg_total / n
    exact = (name_hits == n and arg_total == n and len(pred_calls) == n)
    return name_frac, arg_frac, exact


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):
    gold = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    pred = parse_calls(solution_str or "")
    if not gold:
        # irrelevance example: reward declining, penalize any call
        return 1.0 if pred is None else 0.0
    if pred is None:
        return 0.0
    name_frac, arg_frac, exact = score_calls(pred, gold)
    if BINARY:
        return 1.0 if exact else 0.0
    fmt = 1.0  # parseable
    over_penalty = 0.1 * max(0, len(pred) - len(gold))
    return max(0.0, 0.2 * fmt + 0.3 * name_frac + 0.5 * arg_frac - over_penalty)


if __name__ == "__main__":
    # self-test
    gold = json.dumps([{"name": "get_weather", "arguments": {"city": "Toronto"}},
                       {"name": "get_weather", "arguments": {"city": "Ottawa"}}])
    perfect = '<tool_call>{"name": "get_weather", "arguments": {"city": "Toronto"}}</tool_call>\n' \
              '<tool_call>{"name": "get_weather", "arguments": {"city": "Ottawa"}}</tool_call>'
    half = '<tool_call>{"name": "get_weather", "arguments": {"city": "Toronto"}}</tool_call>'
    junk = "I cannot call tools."
    listy = '[{"name": "get_weather", "arguments": {"city": "toronto"}}, {"name": "get_weather", "arguments": {"city": "OTTAWA"}}]'
    for name, s in [("perfect", perfect), ("half", half), ("junk", junk), ("list+case", listy)]:
        print(f"{name}: {compute_score(None, s, gold):.3f}")
    assert compute_score(None, perfect, gold) == 1.0
    assert compute_score(None, junk, gold) == 0.0
    assert 0.4 < compute_score(None, half, gold) < 0.8
    assert compute_score(None, listy, gold) == 1.0  # case/type-normalized args
    # irrelevance: empty gold — decline is rewarded, calling is not, no crash
    assert compute_score(None, "No tool needed; the answer is 4.", "[]") == 1.0
    assert compute_score(None, half, "[]") == 0.0
    assert score_calls(parse_calls(half), []) == (0.0, 0.0, False)  # no ZeroDivisionError
    print("SELF-TEST: PASS")
