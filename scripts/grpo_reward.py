#!/usr/bin/env python3
"""ToolRL-style verifiable reward for GRPO on tool calling (verl / EasyR1 compatible).

verl custom_reward_function entry point:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

ground_truth: JSON string of gold_calls, e.g. '[{"name": "f", "arguments": {"a": 1}}]'
(from datasets/grpo/*.jsonl). Reward in [0, 1]:
    0.2 * format + 0.3 * name-match + 0.5 * argument-match
Ablation flag BINARY_REWARD=1 switches to exact-match 0/1 (Tool-N1 style).
"""
import json, math, os, re

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


def _json_decoder():
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON number exceeds finite floating-point range")
        return result
    return json.JSONDecoder(object_pairs_hook=unique, parse_constant=invalid_constant, parse_float=finite_float)


def _json_loads(text):
    return _json_decoder().decode(text)


def parse_output(text):
    """Classify calls, no_call, or invalid under the frozen output contract.

    JSON is decoded with a real decoder, including inside <tool_call> blocks.
    Delimiters and thought tags inside argument strings are ordinary values.
    Only leading reasoning blocks are discarded. Empty/malformed output is
    never rewarded as a successful no-call decision.
    """
    text = (text or "").strip()
    invalid = {"status": "invalid", "calls": None}
    while text.startswith("<think>"):
        end = text.find("</think>", len("<think>"))
        if end < 0:
            return invalid
        text = text[end + len("</think>"):].strip()
    if not text:
        return invalid
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    objects = []
    if candidate.startswith(("[", "{")):
        try:
            objects = [_json_loads(candidate)]
        except ValueError:
            return invalid
    elif "<tool_call>" in text or "</tool_call>" in text:
        pos, outside = 0, []
        decoder = _json_decoder()
        while True:
            start = text.find("<tool_call>", pos)
            if start < 0:
                outside.append(text[pos:])
                break
            outside.append(text[pos:start])
            value_start = start + len("<tool_call>")
            while value_start < len(text) and text[value_start].isspace():
                value_start += 1
            try:
                obj, end = decoder.raw_decode(text, value_start)
            except ValueError:
                return invalid
            while end < len(text) and text[end].isspace():
                end += 1
            if not text.startswith("</tool_call>", end):
                return invalid
            objects.append(obj)
            pos = end + len("</tool_call>")
        if not objects or re.search(r"</?(?:think|tool_call|function|parameter)\b|<function=|<parameter=", "".join(outside)):
            return invalid
    elif "<function=" in text:
        functions = XML_FUNCTION_RE.findall(text)
        if len(functions) != text.count("<function=") or len(functions) != text.count("</function>"):
            return invalid
        calls = []
        for name, body in functions:
            args = {}
            params = XML_PARAM_RE.findall(body)
            if (XML_PARAM_RE.sub("", body).strip() or len(params) != body.count("<parameter=")
                    or len(params) != body.count("</parameter>")):
                return invalid
            for key, value in params:
                key = key.strip()
                if not key or key in args:
                    return invalid
                try:
                    value = _json_loads(value.strip())
                except ValueError:
                    value = value.strip()
                args[key] = value
            if not name.strip():
                return invalid
            calls.append({"name": name.strip(), "arguments": args})
        outside = XML_FUNCTION_RE.sub("", text)
        if re.search(r"</?(?:think|tool_call|function|parameter)\b|<function=|<parameter=", outside):
            return invalid
        return {"status": "calls", "calls": calls}
    else:
        if re.search(r"</?(?:think|tool_call|function|parameter)\b|<function=|<parameter=|\[\s*[^\]\n(]+\(\s*(?:\w+\s*=|\))", text):
            return invalid
        return {"status": "no_call", "calls": []}
    calls = []
    for obj in objects:
        for item in obj if isinstance(obj, list) else [obj]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                return invalid
            args = item.get("arguments", item.get("parameters", {}))
            try:
                if isinstance(args, str):
                    args = _json_loads(args)
            except ValueError:
                return invalid
            if not isinstance(args, dict):
                return invalid
            calls.append({"name": item["name"], "arguments": args})
    return {"status": "calls" if calls else "no_call", "calls": calls}


def parse_calls(text):
    """Compatibility wrapper; [] means valid no-call, None means invalid."""
    return parse_output(text)["calls"]


def evaluate_output(text, gold):
    """Shared reward/evaluation semantics; invalid outputs receive zero."""
    parsed = parse_output(text)
    valid = parsed["status"] != "invalid"
    name, arg, exact = score_calls(parsed["calls"], gold) if valid else (0.0, 0.0, False)
    return {"exact": int(exact), "name": name, "arg": arg,
            "literal_exact": int(valid and literal_exact(parsed['calls'], gold)),
            "parsed": int(parsed["status"] == "calls"),
            "valid": int(valid), "status": parsed["status"], "pred": parsed["calls"]}


def literal_exact(pred, gold):
    """Call-multiset equality without case or numeric-string normalization."""
    from collections import Counter
    if pred is None:
        return False
    def key(call):
        return call['name'], json.dumps(call.get('arguments', {}), sort_keys=True,
                                        ensure_ascii=False, allow_nan=False)
    return Counter(map(key, pred)) == Counter(map(key, gold))


def _norm(v):
    if isinstance(v, str):
        s = v.strip().lower()
        for cast in (int, float):
            try:
                result = cast(s)
                if isinstance(result, float) and not math.isfinite(result):
                    continue
                return result
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
    hit = sum(1 for k, v in ga.items() if k in pa and _value_key(pa[k]) == _value_key(v))
    extra = sum(1 for k in pa if k not in ga)
    score = hit / len(ga)
    if extra:
        score *= max(0.0, 1 - 0.25 * extra)
    return 1.0, score


def _value_key(value):
    return json.dumps(_norm(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def _optimal_sum(matrix):
    """Maximum-weight one-to-one assignment (Hungarian method)."""
    if not matrix:
        return 0.0
    n = len(matrix)
    m = max(n, len(matrix[0]))
    cost = [[1 - (row[j] if j < len(row) else 0.0) for j in range(m)] for row in matrix]
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return sum(1 - cost[p[j] - 1][j - 1] for j in range(1, m + 1) if p[j])


def score_calls(pred_calls, gold_calls):
    """Name recall, optimally matched argument credit, normalized call-set EM."""
    from collections import Counter
    if pred_calls is None:
        return 0.0, 0.0, False
    if not gold_calls:
        return (1.0, 1.0, True) if pred_calls == [] else (0.0, 0.0, False)
    gold_names = Counter(c["name"] for c in gold_calls)
    pred_names = Counter(c["name"] for c in pred_calls)
    hits = sum((gold_names & pred_names).values())
    arg_total = 0.0
    for name in gold_names:
        gs = [g for g in gold_calls if g["name"] == name]
        ps = [p for p in pred_calls if p["name"] == name]
        arg_total += _optimal_sum([[match_call(p, g)[1] for p in ps] for g in gs])
    def key(call):
        return call["name"], _value_key(call.get("arguments", {}))
    exact = Counter(map(key, pred_calls)) == Counter(map(key, gold_calls))
    return hits / len(gold_calls), arg_total / len(gold_calls), exact


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None):
    gold = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    pred = parse_calls(solution_str or "")
    if not gold:
        # irrelevance example: reward declining, penalize any call
        return 1.0 if pred == [] else 0.0
    if not pred:
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
