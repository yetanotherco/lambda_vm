#!/usr/bin/env python3
"""G6 host sampler (runs ON THE BOX beside one arm, standard library only).

Every --interval seconds: the cgroup's CPU usage and CFS throttling, /proc/stat's busy and total jiffies, and for every
thread of every process whose argv[0] (or executable) starts with --match, its cumulative on-CPU and runqueue-wait nanoseconds
(/proc/<pid>/task/<tid>/schedstat: time on the cpu, time waiting on a runqueue, timeslices).

A process is matched on its argv[0] alone, the first NUL-separated field of /proc/<pid>/cmdline, never on the whole
command line (a daemon's arguments can hold secrets), or, when argv[0] is relative, on its executable's path
(/proc/<pid>/exe). Only numbers are written, plus each matched thread's own name
(/proc/<pid>/task/<tid>/comm: set by the program or inherited from the thread that created it, at most 15 bytes) once,
kept only if it is made of [A-Za-z0-9_:.-] and written as `?` otherwise. No command line, no environment.

  g6_sampler.py --out <tsv> --match /workspace/lambda_vm-zf-whir/target/release/deps/lambda_vm_prover- [--interval 0.05]
  g6_sampler.py --selftest

Output lines (tab-separated):
  # g6_sampler v2 · interval <s> s · clk_tck <n> · cpus_online <n> · affinity <n> · cpu.max <cgroup quota> · …
  C <t> <cgroup cpu ns> <busy jiffies> <total jiffies> <cgroup nr_throttled> <cgroup throttled ns>   (-1: unreadable)
  T <t> <pid> <tid> <on_ns> <wait_ns>
  N <pid> <tid> <thread name>                  (once per thread, when first seen)
The process list is rescanned every 0.5 s while a matched process lives, and every sample while none does (the
launcher runs the binary twice: a short `--list`, then the test). Stops on SIGTERM or SIGINT (the driver kills it by
PID)."""
import argparse, os, signal, sys, time

STOP = False


def _stop(*_):
    global STOP
    STOP = True


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def cgroup_cpu():
    """(usage ns, nr_throttled, throttled ns) of this cgroup: cgroup v2's cpu.stat, else v1's files; -1 if unreadable."""
    s = _read("/sys/fs/cgroup/cpu.stat")
    if s is not None:
        d = {}
        for line in s.splitlines():
            p = line.split()
            if len(p) == 2 and p[1].lstrip("-").isdigit():
                d[p[0]] = int(p[1])
        if "usage_usec" in d:
            return (d["usage_usec"] * 1000, d.get("nr_throttled", -1),
                    d["throttled_usec"] * 1000 if "throttled_usec" in d else -1)
    usage = nthr = thr = -1
    for p in ("/sys/fs/cgroup/cpuacct/cpuacct.usage", "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage"):
        s = _read(p)
        if s is not None and s.strip().isdigit():
            usage = int(s.strip())
            break
    for p in ("/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"):
        s = _read(p)
        if s is None:
            continue
        d = dict(line.split()[:2] for line in s.splitlines() if len(line.split()) >= 2)
        nthr = int(d.get("nr_throttled", -1))
        thr = int(d.get("throttled_time", -1))
        break
    return usage, nthr, thr


def cpu_max():
    """The cgroup's CPU quota as the kernel states it (`max 100000`, `400000 100000`), or `-`."""
    s = _read("/sys/fs/cgroup/cpu.max")
    if s is not None:
        return s.strip()
    q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    return f"{q.strip()} {p.strip()}" if q is not None and p is not None else "-"


def stat_jiffies():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:9]]
    total = sum(v)
    return total - v[3] - v[4], total


def matching_pids(prefix):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                a0 = f.read().split(b"\0", 1)[0].decode("utf-8", "replace")
        except OSError:
            continue
        if a0.startswith(prefix):
            out.append(int(d))
            continue
        try:
            exe = os.readlink(f"/proc/{d}/exe")
        except OSError:
            continue
        if exe.startswith(prefix):
            out.append(int(d))
    return out


def threads(pid):
    """[(tid, on_ns, wait_ns)] of one process; a thread that exits mid-read is skipped."""
    out = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for t in tids:
        try:
            with open(f"/proc/{pid}/task/{t}/schedstat") as f:
                a = f.read().split()
            out.append((int(t), int(a[0]), int(a[1])))
        except (OSError, ValueError, IndexError):
            continue
    return out


def thread_name(pid, tid):
    """The thread's comm if it is made of [A-Za-z0-9_:.-] (1-15 bytes), else `?`."""
    s = _read(f"/proc/{pid}/task/{tid}/comm")
    s = s.strip() if s is not None else ""
    return s if 0 < len(s) <= 15 and all(c.isascii() and (c.isalnum() or c in "_:.-") for c in s) else "?"


def header(interval):
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = -1
    return (f"# g6_sampler v2 · interval {interval} s · clk_tck {os.sysconf('SC_CLK_TCK')} · cpus_online "
            f"{os.cpu_count()} · affinity {aff} · cpu.max {cpu_max()} · match argv[0] prefix (not printed) · schedstat ns\n")


def run(out_path, prefix, interval, max_seconds=None):
    pids, next_scan, alive, named = [], 0.0, False, set()
    t_end = (time.time() + max_seconds) if max_seconds else None
    with open(out_path, "w", buffering=1 << 20) as out:
        out.write(header(interval))
        while not STOP and (t_end is None or time.time() < t_end):
            t = time.time()
            if t >= next_scan or not alive:
                pids = matching_pids(prefix) if prefix else [os.getpid()]
                next_scan = t + 0.5
            busy, total = stat_jiffies()
            cg, nthr, thr = cgroup_cpu()
            out.write(f"C\t{t:.4f}\t{cg}\t{busy}\t{total}\t{nthr}\t{thr}\n")
            alive = False
            for pid in pids:
                for tid, on, wait in threads(pid):
                    alive = True
                    out.write(f"T\t{t:.4f}\t{pid}\t{tid}\t{on}\t{wait}\n")
                    if (pid, tid) not in named:
                        named.add((pid, tid))
                        out.write(f"N\t{pid}\t{tid}\t{thread_name(pid, tid)}\n")
            slack = interval - (time.time() - t)
            if slack > 0:
                time.sleep(slack)


def selftest():
    """Sample this process for 0.8 s while it burns CPU for 0.6 s: its own on-CPU total must rise by >= 0.3 s, every
    C line must carry six fields, and /proc/stat's busy jiffies must rise."""
    import tempfile, threading
    path = os.path.join(tempfile.mkdtemp(), "selftest.tsv")
    sampler = threading.Thread(target=run, args=(path, "", 0.05, 0.8))
    sampler.start()
    t0 = time.time()
    x = 0
    while time.time() - t0 < 0.6:
        x += 1
    sampler.join()
    on, cl, hdr, names = {}, [], "", []
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if line.startswith("#"):
            hdr = line.strip()
        elif p[0] == "T":
            on.setdefault(int(p[3]), []).append(int(p[4]))
        elif p[0] == "C":
            cl.append(p)
        elif p[0] == "N":
            names.append(p[3])
    grew = sum(max(v) - min(v) for v in on.values()) / 1e9
    six = bool(cl) and all(len(p) == 7 for p in cl)
    busy = int(cl[-1][3]) - int(cl[0][3]) if len(cl) > 1 else 0
    ok = grew >= 0.3 and six and busy > 0 and len(names) == len(on)
    print(f"g6_sampler selftest: {len(on)} thread(s) named {names}, on-CPU grew {grew:.3f} s over the burn · {len(cl)} C lines "
          f"(six fields: {six}; cgroup ns {cl[0][2] if cl else '?'}, throttled {cl[0][6] if cl else '?'}) · box busy "
          f"+{busy} jiffies · {hdr[2:]} : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--match")
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()
    if A.selftest:
        sys.exit(selftest())
    if not A.out or not A.match:
        ap.error("--out and --match are required")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    run(A.out, A.match, A.interval)


if __name__ == "__main__":
    main()
