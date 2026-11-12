"""ZTO-Agent: operator intent → JSON task spec validated by zto/spec.schema.json (schema-constrained decoding).

usage: agent.py --intent "<text>" [--mode full|sparse] [--model llama3.1:8b] [--seed 42] [--temperature T]
                [--out spec.json] [--run-id ID --run-dir DIR]   (timing marks + call log)
                --print-prompt [--mode M]                        (the frozen system prompt + its sha256)
Modes (both reported in Exp-02 as an ablation):
  full    the LLM emits every field (defaults shown in the prompt); constrained by spec.schema.json
  sparse  the LLM emits only the fields the intent states, plus slice_action.action and .target; constrained by
          the sparse schema (spec.schema.json with every other field optional); zto/resolve.py then fills the rest
          from zto/defaults.json deterministically and the result is validated against the full schema
Grounding by enumeration: the system prompt lists the KPM, label-rule and SLICE action catalogues verbatim
(zto/catalog/*.yaml), the schema's admissible ranges and the defaults (zto/defaults.json via resolve.py) —
nothing else. Ollama is asked for JSON constrained by the schema itself (`format`: spec.schema.json, Ollama
structured outputs) and with thinking off (`think`: false). The reply is validated with jsonschema; on failure
the agent re-prompts once with the validation error, then gives up (logged). No max-token cap (orchestration
latency must be the model's own). Every call logs prompt_tokens, completion_tokens, total/load/eval durations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

ZTO = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.dirname(ZTO)
sys.path.insert(0, ZTO)
sys.path.insert(0, R1)
import jsonschema  # noqa: E402

import resolve  # noqa: E402
from common import timing  # noqa: E402

OLLAMA = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11436')
SCHEMA = json.load(open(os.path.join(ZTO, 'spec.schema.json')))
CATALOGS = ('kpm.yaml', 'label_rule.yaml', 'slice_actions.yaml')


def _bounds(node: dict, path: str = '') -> list[str]:
    out = []
    for k, v in (node.get('properties') or {}).items():
        p = f'{path}.{k}' if path else k
        if v.get('type') == 'object':
            out += _bounds(v, p)
        elif 'enum' in v:
            out.append(f'- {p}: one of {json.dumps(v["enum"])}')
        elif 'const' in v:
            out.append(f'- {p}: {json.dumps(v["const"])}')
        elif 'minimum' in v:
            out.append(f'- {p}: {v["type"]} in [{v["minimum"]}, {v["maximum"]}]')
        elif v.get('type') == 'array':
            out.append(f'- {p}: list of {json.dumps(v["items"].get("enum"))}')
    return out


def sparse_schema() -> dict:
    """spec.schema.json with every field optional except slice_action.action and slice_action.target."""
    sch = json.loads(json.dumps(SCHEMA))

    def strip(node):
        node.pop('required', None)
        node.pop('default', None)
        for v in (node.get('properties') or {}).values():
            strip(v)
    strip(sch)
    sch['required'] = ['slice_action']
    sch['properties']['slice_action']['required'] = ['action', 'target']
    sch['title'] += ' — sparse (explicit fields only)'
    return sch


SPARSE_SCHEMA = sparse_schema()
FIELD_CLASSES = json.load(open(os.path.join(ZTO, 'field_classes.json')))


def judge_schema() -> dict:
    """Gate read-back extraction: operator-level fields only (field_classes.json), target required."""
    sch = json.loads(json.dumps(SPARSE_SCHEMA))
    keep = FIELD_CLASSES['operator']
    for top in list(sch['properties']):
        node = sch['properties'][top]
        if node.get('type') == 'object':
            for k in list(node['properties']):
                if f'{top}.{k}' not in keep:
                    node['properties'].pop(k)
            node.pop('required', None)
            if not node['properties']:
                sch['properties'].pop(top)
        elif top not in keep:
            sch['properties'].pop(top)
    sch['required'] = ['slice_action']
    sch['properties']['slice_action']['required'] = ['target']
    sch['title'] = 'Gate read-back extraction (operator-level fields only)'
    return sch


JUDGE_SCHEMA = judge_schema()


INSTRUCTION = {  # the ONLY difference between the two prompts (Exp-02 ablation)
    'full': 'Fill every field. For anything the intent does not state, use the default value.',
    'sparse': ('Output ONLY the fields the intent states explicitly, plus slice_action.action and slice_action.target '
               '(always). Omit every other field: the system fills it from the defaults below. Do not guess values the '
               'intent does not give.'),
}


FIELD_CLASS_SENTENCE = (  # shared verbatim by both orchestrator prompts
    'Operator-level fields (what an operator intent can state): ' + ', '.join(FIELD_CLASSES['operator']) + '. '
    'Engineering-level fields (' + ', '.join(FIELD_CLASSES['engineering']) + ') are set by the system: keep their '
    'default values.\n')


def system_prompt(mode: str = 'full') -> str:
    cats = '\n\n'.join(f'### zto/catalog/{c}\n{open(os.path.join(ZTO, "catalog", c)).read().strip()}' for c in CATALOGS)
    if mode == 'judge':  # gate read-back: independent extraction of operator-level fields from the intent text only
        return ('You read an operator intent for a congestion-prediction xApp in an O-RAN cell (FlexRIC) and extract '
                'ONLY the operator-level fields it states explicitly: ' + ', '.join(FIELD_CLASSES['operator']) + '. '
                'Reply with ONLY the JSON object; it must validate against the schema you are given. Omit every field '
                'the intent does not state. slice_action.target is always present: use "unspecified" if the intent '
                'does not say who should benefit. Use "unspecified" for label_rule.op if the direction is not stated.\n\n'
                '## Admissible values (spec.schema.json, operator-level fields)\n' + '\n'.join(_bounds(JUDGE_SCHEMA))
                + '\n\n## Catalogues\n' + cats)
    return ('You translate an operator intent into the JSON task specification of a congestion-prediction xApp for an '
            'O-RAN cell (FlexRIC). Reply with ONLY the JSON object; it must validate against the schema you are given.\n'
            'Use only names from the catalogues below. Never invent fields.\n'
            + INSTRUCTION[mode] + '\n'
            'slice_action.target has no default: if the intent does not say who should benefit, use "unspecified". '
            'If it does not say whether congestion means above or below the threshold, label_rule.op is "unspecified".\n'
            + FIELD_CLASS_SENTENCE + '\n'
            '## Admissible values (spec.schema.json)\n' + '\n'.join(_bounds(SCHEMA)) + '\n\n'
            '## Defaults (zto/defaults.json)\n' + json.dumps(resolve.resolve({}), indent=1) + '\n\n'
            '## Catalogues\n' + cats)


def prompt_sha256(mode: str) -> str:
    import hashlib
    return hashlib.sha256(system_prompt(mode).encode()).hexdigest()


def call(model: str, messages: list[dict], seed: int, temperature: float | None, schema: dict = SCHEMA) -> tuple[dict, float]:
    opts = {'seed': seed} | ({} if temperature is None else {'temperature': temperature})
    body = {'model': model, 'messages': messages, 'stream': False, 'think': False, 'format': schema,
            'keep_alive': -1, 'options': opts}
    t0 = time.monotonic()
    req = urllib.request.Request(f'{OLLAMA}/api/chat', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return r, time.monotonic() - t0


def errors(spec, schema: dict = SCHEMA) -> list[str]:
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(spec)]


def run(intent: str, *, model: str = 'llama3.1:8b', seed: int = 42, temperature: float | None = None,
        mode: str = 'full', tm: timing.Timer | None = None) -> dict:
    """→ {spec (full, resolved for sparse)|None, sparse_spec, valid, attempts, calls: [token/latency log], error}."""
    schema = {'full': SCHEMA, 'sparse': SPARSE_SCHEMA, 'judge': JUDGE_SCHEMA}[mode]
    lock = json.load(open(os.path.join(ZTO, 'prompts', 'lock.json')))['sha256'][mode]
    if prompt_sha256(mode) != lock and os.environ.get('R1_ALLOW_UNFROZEN_PROMPT') != '1':
        raise RuntimeError(f'{mode} prompt differs from the frozen one (prompts/lock.json); '
                           'a catalogue, defaults or prompt edit after the freeze invalidates Exp-02')
    msgs = [{'role': 'system', 'content': system_prompt(mode)}, {'role': 'user', 'content': intent}]
    calls, spec, errs = [], None, ['no reply']
    for attempt in (1, 2):
        tm and tm.mark('agent_request', attempt=attempt, model=model)
        r, wall = call(model, msgs, seed, temperature, schema)
        tm and tm.mark('agent_reply', attempt=attempt, model=model)
        content = r['message']['content']
        calls.append({'attempt': attempt, 'prompt_tokens': r.get('prompt_eval_count'),
                      'completion_tokens': r.get('eval_count'), 'total_duration_ms': r.get('total_duration', 0) / 1e6,
                      'load_duration_ms': r.get('load_duration', 0) / 1e6, 'eval_duration_ms': r.get('eval_duration', 0) / 1e6,
                      'prompt_eval_duration_ms': r.get('prompt_eval_duration', 0) / 1e6, 'wall_s': round(wall, 3)})
        try:
            spec = json.loads(content)
            errs = errors(spec, schema)
        except ValueError as exc:
            spec, errs = None, [f'not JSON: {exc}']
        if not errs:
            break
        if attempt == 1:  # re-prompt once with the validation error
            msgs += [{'role': 'assistant', 'content': content},
                     {'role': 'user', 'content': 'That JSON does not validate: ' + '; '.join(errs)
                      + '. Reply with the corrected JSON object only.'}]
    sparse = spec if mode in ('sparse', 'judge') else None
    if not errs and mode == 'sparse':  # deterministic fill from zto/defaults.json, then the full schema
        spec = resolve.resolve(spec)
        errs = errors(spec)
    valid = not errs
    tm and tm.mark('agent_spec_valid' if valid else 'agent_gave_up', model=model, mode=mode)
    return {'spec': spec if valid else None, 'raw_spec': spec, 'sparse_spec': sparse, 'valid': valid,
            'attempts': len(calls), 'calls': calls, 'error': None if valid else errs, 'model': model, 'seed': seed,
            'temperature': temperature, 'think': False, 'mode': mode, 'prompt_sha256': prompt_sha256(mode)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intent', required=True)
    ap.add_argument('--model', default='llama3.1:8b')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--temperature', type=float)
    ap.add_argument('--out')
    ap.add_argument('--run-id')
    ap.add_argument('--run-dir')
    ap.add_argument('--mode', choices=('full', 'sparse', 'judge'), default='full')
    ap.add_argument('--print-prompt', action='store_true')
    a = ap.parse_args()
    if a.print_prompt:
        print(system_prompt(a.mode))
        print(f'\n# sha256 {prompt_sha256(a.mode)}', file=sys.stderr)
        return
    tm = timing.Timer(a.run_id, a.run_dir) if a.run_id and a.run_dir else None
    res = run(a.intent, model=a.model, seed=a.seed, temperature=a.temperature, mode=a.mode, tm=tm)
    if a.out and res['valid']:
        open(a.out, 'w').write(json.dumps(res['spec'], indent=2) + '\n')
    if a.run_dir:
        with open(os.path.join(a.run_dir, 'agent_calls.jsonl'), 'a') as fh:
            fh.write(json.dumps(res) + '\n')
    print(json.dumps(res, indent=1))
    sys.exit(0 if res['valid'] else 1)


if __name__ == '__main__':
    main()
