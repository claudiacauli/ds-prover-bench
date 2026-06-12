import argparse
import codecs
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Final

WORKSPACE: Final = Path("/workspace")
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]

MINI_F2F_PATH: Final = PROJECT_ROOT / "data" / "minif2f.jsonl"
DEEPSEEK_PROVER_V2_7B: Final = PROJECT_ROOT / "models" / "DeepSeek-Prover-V2-7B"
LAKE_PATH: Final = WORKSPACE / "elan" / "bin" / "lake"
MATHLIB_DIR: Final = WORKSPACE / "DeepSeek-Prover-V1.5" / "mathlib4"
LEAN_REPL_LOG: Final = PROJECT_ROOT / "lean_repl.log"


class Color:
    _enabled = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    GREEN: Final = "\033[92m" if _enabled else ""
    RED: Final = "\033[91m" if _enabled else ""
    RESET: Final = "\033[0m" if _enabled else ""


class Split(StrEnum):
    TEST = "test"
    VALID = "valid"


@dataclass(frozen=True)
class RunConfig:
    model_path: str
    model_revision: str
    n: int
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    prompt_style: str  # cot or non_cot
    max_heartbeats: int
    mathlib_commit: str
    lean_version: str
    validity: str
    split: Split
    verify_timeout_s: int
    max_model_len: int


@dataclass(frozen=True)
class Minif2fEntry:
    name: str
    split: Split
    informal_prefix: str
    formal_statement: str
    goal: str
    header: str


class Completion(Enum):
    TOTAL = "Total"
    PARTIAL = "Partial"


@dataclass(frozen=True)
class ProblemResult:
    config_hash: str
    problem_name: str
    formal_statement_hash: str
    model_completions: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return (self.config_hash, self.problem_name)


@dataclass(frozen=True)
class EvalResult:
    evaluated: int
    total: int
    solved: int

    @property
    def completion(self) -> Completion:
        return Completion.TOTAL if self.evaluated == self.total else Completion.PARTIAL

    @property
    def pass_at_k(self) -> float:
        return self.solved / self.evaluated if self.evaluated else 0.0


def send_command(repl: Any, command: dict[str, Any], timeout: int) -> dict[str, Any]:
    if repl.poll() is not None:
        raise RuntimeError(
            f"Lean REPL is dead (exit code {repl.returncode}); check lean_repl.log and restart it."
        )
    fd = repl.stdout.fileno()
    while select.select([repl.stdout], [], [], 0)[0]:
        if os.read(fd, 65536) == b"":
            break
    repl.stdin.write((json.dumps(command) + "\r\n\r\n").encode())
    repl.stdin.flush()
    json_decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    text = ""
    deadline = time.time() + timeout
    while True:
        remaining_time = deadline - time.time()
        if remaining_time <= 0 or not select.select([repl.stdout], [], [], remaining_time)[0]:
            raise TimeoutError(f"verification exceeded {timeout}s")
        chunk = os.read(fd, 65536)
        if chunk == b"":
            raise RuntimeError("Lean REPL closed its output (died mid-command)")
        text += utf8.decode(chunk)
        try:
            return json_decoder.raw_decode(text.lstrip())[0]
        except json.JSONDecodeError:
            continue


class LeanRepl:
    proc: subprocess.Popen[bytes]

    def __init__(self, header: str):
        self.header = header
        self.start()

    def _spawn(self):
        self._close_log()
        self._log = open(LEAN_REPL_LOG, "a")
        self.proc = subprocess.Popen(
            ["stdbuf", "-o0", LAKE_PATH, "exe", "repl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            cwd=MATHLIB_DIR,
            start_new_session=True,
        )

    def _close_log(self):
        log = getattr(self, "_log", None)
        if log is not None:
            log.close()

    def _kill(self):
        proc = getattr(self, "proc", None)
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            for pipe in (proc.stdin, proc.stdout):
                if pipe is not None:
                    pipe.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def start(self, retries: int = 3):
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                self._spawn()
                send_command(self.proc, {"cmd": self.header}, 600)
                return
            except Exception as e:
                last_err = e
                print(f"   repl (re)start attempt {attempt}/{retries} failed: {e}")
                self._kill()
                if attempt < retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"REPL failed to start after {retries} attempts") from last_err

    def restart(self):
        self._kill()
        self.start()

    def close(self):
        self._kill()
        self._close_log()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        self.close()


def _mathlib_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(MATHLIB_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def _lean_version() -> str:
    return (MATHLIB_DIR / "lean-toolchain").read_text().strip()


def build_config(model_path: Path | None = None) -> RunConfig:
    return RunConfig(
        model_path=str(
            Path(model_path if model_path is not None else DEEPSEEK_PROVER_V2_7B).resolve()
        ),
        model_revision="local",
        n=4,
        temperature=1.0,
        top_p=0.95,
        max_tokens=1024,
        seed=0,
        prompt_style="non_cot",
        max_heartbeats=400_000,
        mathlib_commit=_mathlib_commit(),
        lean_version=_lean_version(),
        validity="no_error_no_sorry",
        split=Split.TEST,
        verify_timeout_s=300,
        max_model_len=4096,
    )


def config_hash(cfg: RunConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def load_minif2f() -> list[Minif2fEntry]:
    with open(MINI_F2F_PATH) as file:
        tests: list[Minif2fEntry] = []
        for line in file:
            entry: dict[str, str] = json.loads(line)
            tests.append(
                Minif2fEntry(
                    name=entry["name"],
                    split=Split(entry["split"]),
                    informal_prefix=entry["informal_prefix"],
                    formal_statement=entry["formal_statement"],
                    goal=entry["goal"],
                    header=entry["header"],
                )
            )
    return tests


def build_prompt(entry: Minif2fEntry, cfg: RunConfig) -> str:
    if cfg.prompt_style != "non_cot":
        raise ValueError(f"unsupported prompt_style: {cfg.prompt_style!r}")
    return (
        f"Complete the following Lean 4 code:\n\n```lean4\n{entry.header}"
        f"{entry.informal_prefix}{entry.formal_statement}"
    )


def call_llm(preloaded_llm: Any, prompt: str, test_n: int, cfg: RunConfig) -> list[dict[str, str]]:
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        n=cfg.n,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
    )
    resp = preloaded_llm.generate(prompt, sampling_params=params, use_tqdm=False)
    print(f" • [Test #{test_n}] Generated k={cfg.n} proof attempts", end=" ", flush=True)
    return resp[0].outputs


def process_resp(entry: Minif2fEntry, resp_outputs: list[Any], cfg: RunConfig) -> ProblemResult:
    completions: list[str] = []
    for output in resp_outputs:
        completions.append(output.text)
    return ProblemResult(
        config_hash=config_hash(cfg),
        problem_name=entry.name,
        formal_statement_hash=hashlib.sha256(entry.formal_statement.encode("utf-8")).hexdigest()[
            :16
        ],
        model_completions=tuple(completions),
    )


def build_lean(entry: Minif2fEntry, resp_text: str, cfg: RunConfig) -> str:
    proof: str = resp_text.split("```")[0]
    return f"set_option maxHeartbeats {cfg.max_heartbeats} in\n{entry.formal_statement}{proof}"


def is_proof_valid(resp: dict[str, Any] | None) -> bool:
    # Validity is recorded only for hashing (and provenance).
    # There is no logic other than no_error_no_sorry
    if resp is None:
        return False
    has_error = [m for m in resp.get("messages", []) if m.get("severity") == "error"]
    has_sorry = bool(resp.get("sorries"))
    return not has_error and not has_sorry


def is_problem_solved(
    entry: Minif2fEntry, test_n: int, preloaded_llm: Any, lean_repl: LeanRepl, cfg: RunConfig
) -> bool:
    prompt = build_prompt(entry, cfg)
    resp_outputs = call_llm(preloaded_llm, prompt, test_n, cfg)
    problem_result = process_resp(entry, resp_outputs, cfg)
    for completion in problem_result.model_completions:
        lean = build_lean(entry, completion, cfg)
        try:
            resp = send_command(
                lean_repl.proc, {"cmd": lean, "env": 0}, timeout=cfg.verify_timeout_s
            )
        except (TimeoutError, RuntimeError):
            print(" ⏱ Timeout or REPL killed", end=" ", flush=True)
            lean_repl.restart()
            continue
        if is_proof_valid(resp):
            print(f" {Color.GREEN}✔{Color.RESET} VALID ")
            return True
    print(f" {Color.RED}✗{Color.RESET} INVALID")
    return False


def load_already_solved(checkpoint: Path) -> dict[str, bool]:
    # Caches everything from a previous run; whatever was already computed, it's now skipped.
    # This means that to re-compute something, one needs to manually delete the checkpoint
    # file, and force it to regenerate.
    done: dict[str, bool] = {}
    if not os.path.exists(checkpoint):
        return done
    with open(checkpoint) as file:
        for line in file:
            entry = json.loads(line)
            done[entry["name"]] = entry["solved"]
    return done


def evaluate(
    tests: list[Minif2fEntry],
    preloaded_llm: Any,
    lean_repl: LeanRepl,
    cfg: RunConfig,
    checkpoint: Path,
) -> EvalResult:
    already_solved = load_already_solved(checkpoint)
    evaluated = solved = 0
    for i, entry in enumerate(tests):
        name = entry.name
        if name in already_solved:
            verdict = already_solved[name]
            print(f" • [Test #{i}] Recovered from checkpoint", end=" ", flush=True)
            print(
                f" {Color.GREEN}✔{Color.RESET} VALID "
                if verdict
                else f" {Color.RED}✗{Color.RESET} INVALID"
            )
        else:
            try:
                verdict = is_problem_solved(entry, i, preloaded_llm, lean_repl, cfg)
            except RuntimeError as e:
                print(f"\n ! Stopping early at #{i}: {e}")
                print(f"  Progress saved in {checkpoint.name} - re-run to resume.")
                break
            with open(checkpoint, "a") as f:
                f.write(json.dumps({"name": name, "solved": verdict}) + "\n")
        evaluated += 1
        solved += int(verdict)
    return EvalResult(evaluated=evaluated, total=len(tests), solved=solved)


def shared_header(tests: list[Minif2fEntry]) -> str:
    headers = {t.header for t in tests}
    if len(headers) != 1:
        raise ValueError(
            f"LeanRepl builds one env from one header, but there are {len(headers)} "
            f"distinct headers. One env per header is not supported."
        )
    return headers.pop()


def main(
    preloaded_llm: Any,
    lean_repl: LeanRepl,
    cfg: RunConfig,
    tests: list[Minif2fEntry],
    resume: Path | None = None,
) -> None:
    if resume is None:
        # fresh run
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint = PROJECT_ROOT / f"gen_checkpoint_{config_hash(cfg)}_{ts}.jsonl"
        print(f"• Fresh run on {len(tests)} '{cfg.split}' tests → {checkpoint.name}")
    else:
        # checkpoint run
        if not resume.exists():
            raise FileNotFoundError(f"--resume target not found: {resume}")
        if config_hash(cfg) not in resume.name:
            raise ValueError(f"{resume.name} is from a different config than the current one")
        checkpoint = resume
        done = load_already_solved(checkpoint)
        print(f"• Resuming {checkpoint.name}: {len(done)}/{len(tests)} already recorded")

    r = evaluate(tests, preloaded_llm, lean_repl, cfg, checkpoint)
    if r.completion is Completion.TOTAL:
        print(f"• [Total] pass@{cfg.n} = {r.pass_at_k:.3f} ({r.solved}/{r.total})")
    else:
        print(
            f"• [Partial] pass@{cfg.n} = {r.pass_at_k:.3f} "
            f"({r.solved}/{r.evaluated} solved - only {r.evaluated}/{r.total})"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "path to the DeepSeek-Prover-V2-7B model (default: <repo>/models/DeepSeek-Prover-V2-7B)"
        ),
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--resume", type=Path, default=None, metavar="CHECKPOINT", help="resume existing checkpoint"
    )
    g.add_argument("--resume-latest", action="store_true", help="resume latest checkpoint")
    args = parser.parse_args()

    cfg = build_config(model_path=args.model_path)
    if not Path(cfg.model_path).exists():
        sys.exit(
            f"model not found at {cfg.model_path}. "
            "Run ./install.sh or pass --model-path with a valid path"
        )

    tests: list[Minif2fEntry] = [t for t in load_minif2f() if t.split == cfg.split]
    header = shared_header(tests)
    if args.resume_latest:
        matches = sorted(PROJECT_ROOT.glob(f"gen_checkpoint_{config_hash(cfg)}_*.jsonl"))
        resume = matches[-1] if matches else None
    else:
        resume = args.resume

    from vllm import LLM

    preloaded_llm = LLM(model=cfg.model_path, max_model_len=cfg.max_model_len)

    with LeanRepl(header) as lean_repl:
        main(preloaded_llm, lean_repl, cfg, tests, resume=resume)
