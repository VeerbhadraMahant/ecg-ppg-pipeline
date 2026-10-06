"""Label-efficiency sweep (updates.md analysis 6.7): fine-tune with 10/25/50/100 %
of the training records, comparing scratch vs each pretraining objective,
fine-tuned vs frozen encoders, for fusion (cross_attention) vs single-signal
models. Runs src.train as parallel subprocesses (the models are small, the GPU
has headroom).

    python scripts/run_label_efficiency.py --workers 4 --seeds 3
"""
import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, log, tries=4):
    """Retry: Windows application control occasionally blocks a native extension
    when many processes import it at once; training itself resumes per fold."""
    import time

    rc = 1
    for k in range(tries):
        with open(log, "a" if k else "w") as f:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if rc == 0:
            return rc
        time.sleep(10 * (k + 1))
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="cross_attention,ecg_only,ppg_only")
    ap.add_argument("--fractions", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--inits", default="scratch,contrastive,crossmodal,masked")
    ap.add_argument("--processed", default="data/processed/challenge2015_windows.npz")
    ap.add_argument("--prefix", default="labeleff")
    ap.add_argument("--width-override", type=float, default=None)
    args = ap.parse_args()

    jobs = []
    for init, frac in itertools.product(args.inits.split(","), args.fractions.split(",")):
        modes = ["finetune"] if init == "scratch" else ["finetune", "frozen"]
        for mode in modes:
            name = f"{args.prefix}/{init}_{mode}_f{frac}"
            cmd = [sys.executable, "-u", "-m", "src.train", "--variant", args.variants, "--seeds", str(args.seeds),
                   "--label-fraction", frac, "--run-name", name, "--processed", args.processed]
            if args.width_override:
                cmd += ["--width-override", str(args.width_override)]
            if init != "scratch":
                cmd += ["--init-from", f"runs/pretrain/{init}.pt"]
                if mode == "frozen":
                    cmd += ["--freeze-encoders"]
            log = ROOT / "runs" / (name.replace("/", "_") + ".log")
            jobs.append((cmd, log, name))
    print(f"{len(jobs)} runs")
    with ThreadPoolExecutor(args.workers) as ex:
        for (cmd, log, name), rc in zip(jobs, ex.map(lambda j: run(j[0], j[1]), jobs)):
            print(f"{name}: exit {rc}", flush=True)


if __name__ == "__main__":
    main()
