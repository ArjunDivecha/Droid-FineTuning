#!/usr/bin/env python3
"""
Monitor a debug_runs/<run>/events.jsonl and mirror key summaries
to a consolidated plain-text log file for easy viewing.

Usage:
  python3 tools/monitor_debug_events.py \
    --debug-root debug_runs/voice_pilot \
    --out logs/voice_pilot.out \
    --follow

If --follow is set, the script will wait for new lines and append them.
"""
import argparse, json, time
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug-root', required=True, help='Path containing timestamped runs')
    ap.add_argument('--out', required=True, help='Output consolidated log file')
    ap.add_argument('--follow', action='store_true', help='Follow for new lines')
    return ap.parse_args()


def latest_run_path(root: Path) -> Path:
    runs = [p for p in root.glob('*') if p.is_dir()]
    if not runs:
        return root
    return sorted(runs)[-1]


def format_event(obj: dict) -> str:
    ev = obj.get('event')
    ts = obj.get('timestamp')
    if ev == 'train_report':
        it = obj.get('iteration')
        loss = obj.get('train_loss')
        kl = obj.get('train_kl') or obj.get('kl')
        itps = obj.get('iterations_per_second')
        return f"[TRAIN] iter={it} loss={loss:.6f} kl={kl:.6f if isinstance(kl,(int,float)) else kl} it/s={itps:.3f}"
    if ev == 'val_report':
        it = obj.get('iteration')
        v = obj.get('val_loss')
        t = obj.get('val_time')
        return f"[EVAL]  iter={it} val_loss={v:.6f} time={t:.2f}s"
    if ev == 'checkpoint':
        it = obj.get('step')
        ck = obj.get('checkpoint') or obj.get('adapter')
        return f"[CKPT]  iter={it} saved={ck}"
    if ev == 'empty_completion_tokens':
        step = obj.get('step')
        idx = obj.get('indices')
        return f"[WARN]  empty_tokens at step={step} indices={idx}"
    return ''


def tail_events(run_dir: Path, out_file: Path, follow: bool):
    events = run_dir / 'events.jsonl'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not events.exists():
        print(f"No events file yet at {events}")
        return
    # Remember file position
    pos = 0
    with events.open('r', encoding='utf-8') as f_in, out_file.open('a', encoding='utf-8') as f_out:
        # Fast-forward existing content
        for line in f_in:
            try:
                rec = json.loads(line)
                msg = format_event(rec)
                if msg:
                    f_out.write(msg + "\n")
                    f_out.flush()
            except Exception:
                continue
        pos = f_in.tell()
        if not follow:
            return
        # Follow
        while True:
            time.sleep(1.0)
            f_in.seek(pos)
            new = f_in.read()
            if not new:
                continue
            pos = f_in.tell()
            for line in new.splitlines():
                try:
                    rec = json.loads(line)
                    msg = format_event(rec)
                    if msg:
                        f_out.write(msg + "\n")
                        f_out.flush()
                except Exception:
                    continue


def main():
    args = parse_args()
    root = Path(args.debug_root)
    run_dir = latest_run_path(root)
    tail_events(run_dir, Path(args.out), args.follow)

if __name__ == '__main__':
    main()

