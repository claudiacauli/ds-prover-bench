import json
import subprocess
from vllm import LLM, SamplingParams

MINI_F2F_PATH = "/workspace/ds-prover-bench/data/minif2f.jsonl"
DEEKSEEK_PROVER_V2_7B = "/workspace/models/DeepSeek-Prover-V2-7B"
LAKE_PATH="/workspace/elan/bin/lake"
MATHLIB_DIR="/workspace/DeepSeek-Prover-V1.5/mathlib4"

def load_minif2f():
    with open(MINI_F2F_PATH) as file:
        tests = []
        for line in file:
            entry = json.loads(line)
            tests.append(entry)
    return tests
    

def build_prompt(entry):
    return f"Complete the following Lean 4 code:\n\n```lean4\n{entry.get('header')}{entry.get('informal_prefix')}{entry.get('formal_statement')}"


def call_llm(prompt):
    llm = LLM(model=DEEKSEEK_PROVER_V2_7B, max_model_len=4096)
    params = SamplingParams(temperature=1.0, top_p=0.95, n=4, max_tokens=1024)
    resp = llm.generate(prompt, sampling_params=params)
    print()
    print(resp[0].outputs[0].text)
    print(resp[0].outputs[1].text)
    print(resp[0].outputs[2].text)
    print(resp[0].outputs[3].text)
    print()
    return resp[0].outputs


def process_resp(entry, resp_outputs):
    lean_files = []
    for output in resp_outputs:
        lean_files.append(build_lean(entry, output.text))
    return lean_files


def build_lean(entry, resp_text):
    proof = resp_text.split('```')[0]
    return f"{entry.get('formal_statement')}{proof}"


def parse_responses(stdout):
    decoder = json.JSONDecoder()
    responses, i = [], 0
    while i < len(stdout):
        while i < len(stdout) and stdout[i].isspace():  # skip blank space between objects
            i += 1
        if i >= len(stdout):
            break
        obj, end = decoder.raw_decode(stdout, i)         # parse ONE object
        responses.append(obj)
        i = end                                          # continue after it
    return responses


def verify_all(entry, lean_bodies):
    commands = [{"cmd": entry.get('header')}]
    for body in lean_bodies:
        commands.append({"cmd": body, "env": 0})
    payload = "".join(json.dumps(c) + "\r\n\r\n" for c in commands)
    results = subprocess.run(
        [LAKE_PATH, "exe", "repl"],
        input=payload,
        cwd=MATHLIB_DIR,
        text=True,
        capture_output=True,
        timeout=300
    )
    responses = parse_responses(results.stdout)
    return responses[1:]


def is_proof_valid(resp):
    error_msgs = [m for m in resp.get('messages', []) if m['severity'] == 'error']
    return not error_msgs


if __name__ == "__main__": 
    dic_tests = load_minif2f()
    print(f" - {len(dic_tests)} tests loaded from the miniF2F dataset.")
    one_prompt = build_prompt(dic_tests[0])
    resp_outputs = call_llm(one_prompt)
    lean_files = process_resp(dic_tests[0], resp_outputs)
    results = verify_all(dic_tests[0], lean_files)
    for resp in results:
        print(" - ✔ Valid Proof" if is_proof_valid(resp) else " - ✗ INVALID Proof")
