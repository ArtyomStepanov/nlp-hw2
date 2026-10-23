"""Bounded benchmark window: pause at low GPU memory, always resume training."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "artifacts/attention_benchmark/window.json"
REPORT = Path(__file__).resolve().parent
BATCH_CHECK = "--batch-check" in sys.argv
if BATCH_CHECK:
    STATE = STATE.with_name("batch_window.json")


def identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except FileNotFoundError:
        return None


def resume(state):
    if identity(state["train_pid"]) == state["train_identity"]:
        try:
            os.kill(state["train_pid"], signal.SIGCONT)
        except ProcessLookupError:
            pass  # process exited between the identity check and signal


def stop_benchmark(state):
    pid = state.get("benchmark_pid")
    if pid and identity(pid) == state.get("benchmark_identity"):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def watchdog():
    while True:
        state = json.loads(STATE.read_text())
        if state.get("finished"):
            return
        if identity(state["controller_pid"]) != state["controller_identity"] or time.time() > state["deadline"]:
            try:
                stop_benchmark(state)
            finally:
                resume(state)
            print("WATCHDOG_RESUMED_TRAINING", flush=True)
            return
        time.sleep(5)


def save(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


def main():
    pid = int(sys.argv[1])
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
    assert "train_distributed.py" in cmdline and "transformer_003" in cmdline, cmdline
    state = dict(train_pid=pid, train_identity=identity(pid), controller_pid=os.getpid(),
                 controller_identity=identity(os.getpid()), deadline=time.time()+2400, results=[])
    save(state)
    subprocess.Popen([sys.executable, __file__, "watchdog"] + (["--batch-check"] if BATCH_CHECK else []), start_new_session=True)
    try:
        print("WAITING_FOR_VALIDATION_MEMORY", flush=True)
        until = time.time()+650
        while True:
            assert identity(pid) == state["train_identity"], "Training exited before benchmark window"
            output = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"
            ], text=True)
            memories = {int(p): int(m) for p,m in (line.split(",") for line in output.strip().splitlines())}
            if 0 < memories.get(pid, 99999) < 12000:
                os.kill(pid, signal.SIGSTOP)
                state["paused_at"] = time.time()
                state["paused_memory_mib"] = memories[pid]
                save(state)
                print("PAUSED", json.dumps(state), flush=True)
                time.sleep(3)  # drain already submitted kernels
                break
            if time.time() > until:
                raise TimeoutError("No low-memory validation window within 650s; training unchanged")
            time.sleep(2)
        env = os.environ | {
            "PYTHONPATH": str(ROOT/"artifacts/attention_benchmark/deps012")+":"+str(ROOT/"artifacts/attention_benchmark/deps"),
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "USE_HUB_KERNELS": "NO",
            "TOKENIZERS_PARALLELISM": "false", "TORCHINDUCTOR_COMPILE_THREADS": "4",
        }
        jobs = [(backend + ("_compiled" if compiled else "_eager"), backend, compiled, [])
                for compiled in (False, True) for backend in ("sdpa", "fa2", "fa3")]
        if BATCH_CHECK:
            jobs = [("sdpa_batch256_compiled", "sdpa", True, ["--batch", "256"]),
                    ("sdpa_batch256_stress", "sdpa", True, ["--batch", "256", "--stress", "--repeats", "1"]),
                    ("sdpa_batch128_control", "sdpa", True, [])]
        for name, backend, compile_mode, extra in jobs:
                args = [sys.executable, str(REPORT/"compare_attention.py"), "--backend", backend,
                        "--result", str(REPORT/(name+".json"))] + extra
                if compile_mode:
                    args.append("--compile")
                print("START",name,flush=True)
                with (ROOT/"artifacts/attention_benchmark"/(name+".log")).open("w") as log:
                    child = subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(benchmark_pid=child.pid,benchmark_identity=identity(child.pid))
                    save(state)
                    try:
                        code = child.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
                        code = "timeout"
                state["results"].append(dict(variant=name,exit_code=code))
                state.pop("benchmark_pid",None)
                save(state)
                print("DONE",name,code,flush=True)
    finally:
        try:
            stop_benchmark(state)
        finally:
            resume(state)
        state.update(finished=True,resumed_at=time.time())
        save(state)
        (REPORT/STATE.name).write_text(json.dumps(state,indent=2))
        print("RESUMED_TRAINING",flush=True)


if __name__ == "__main__":
    watchdog() if sys.argv[1] == "watchdog" else main()
