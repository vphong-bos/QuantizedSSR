import os
import signal
import subprocess

def run_script(cmd, cwd="/kaggle/working/QuantizedSSR", extra_env=None, label=None):
    """
    Generic subprocess runner with streaming logs and safe shutdown.

    Args:
        cmd (list): Command list (e.g. ["python", "script.py", "--arg", "val"])
        cwd (str): Working directory
        extra_env (dict, optional): env vars
        label (str, optional): Name for logging
    """

    # ---- Environment ----
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    if extra_env:
        env.update(extra_env)

    if label:
        print(f"\n[INFO] Running: {label}")
    print("[INFO] Command:", " ".join(cmd))

    # ---- Start process ----
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid
    )

    # ---- Stream logs ----
    try:
        for line in process.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        print("\n[INFO] Stopping process...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    # ---- Cleanup ----
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[WARN] Force killing process...")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    if process.returncode != 0:
        raise RuntimeError(f"Process failed with exit code {process.returncode}")