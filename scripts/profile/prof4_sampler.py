#!/usr/bin/env python3
"""prof4.sh's host sampler and memory watchdog: one line every PERIOD seconds until it is killed.

    prof4_sampler.py <binary path> <out.tsv> <watchdog file> [period_s] [floor_mib]

Columns: unix time · cgroup memory bytes · cgroup anon bytes · cgroup CPU usec · the prover's pid · its VmRSS kB · its
utime+stime clock ticks · its threads · nsys's VmRSS kB (summed) · MemAvailable kB. The prover is the process whose
/proc/<pid>/exe is <binary path> (under nsys or ncu it is their child); -1 when absent.

Watchdog: when MemAvailable falls below floor_mib (default 1536), the prover gets SIGTERM, then SIGKILL 10 s later, and
the watchdog file says so. The profiler then writes what it has, and the next run waits for memory to come back.
"""
import os
import signal
import sys
import time

BIN, OUT, WD = sys.argv[1], sys.argv[2], sys.argv[3]
PERIOD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
FLOOR_KB = int(sys.argv[5] if len(sys.argv) > 5 else 1536) * 1024
CG = "/sys/fs/cgroup"


def rd(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ""


def cg_cpu_usec():
    for line in rd(f"{CG}/cpu.stat").splitlines():
        if line.startswith("usage_usec "):
            return int(line.split()[1])
    return -1


def cg_mem():
    v = rd(f"{CG}/memory.current").strip()
    return int(v) if v.isdigit() else -1


def cg_anon():
    for line in rd(f"{CG}/memory.stat").splitlines():
        k, _, v = line.partition(" ")
        if k == "anon":
            return int(v)
    return -1


def mem_available_kb():
    for line in rd("/proc/meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return -1


def exe_pids():
    mine, prof = [], []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            e = os.readlink(f"/proc/{d}/exe")
        except OSError:
            continue
        if e == BIN:
            mine.append(int(d))
        elif os.path.basename(e).startswith(("nsys", "ncu")) or os.path.basename(e) in ("QdstrmImporter", "TargetProcess"):
            prof.append(int(d))
    return mine, prof


def status_kb(pid, key):
    for line in rd(f"/proc/{pid}/status").splitlines():
        if line.startswith(key):
            return int(line.split()[1])
    return -1


def ticks(pid):
    s = rd(f"/proc/{pid}/stat")
    if not s:
        return -1, -1
    rest = s[s.rfind(")") + 2:].split()
    if rest[0] in ("Z", "X"):
        return -1, -1
    return int(rest[11]) + int(rest[12]), int(rest[17])


fired = False
with open(OUT, "w", buffering=1) as f:
    f.write("t\tcg_mem\tcg_anon\tcg_cpu_usec\tpid\trss_kb\tticks\tthreads\tprof_rss_kb\tmemavail_kb\n")
    n = 0
    mine, prof = [], []
    while True:
        if n % 4 == 0 or not mine:
            mine, prof = exe_pids()
        n += 1
        pid = mine[0] if mine else -1
        rss = status_kb(pid, "VmRSS:") if pid > 0 else -1
        tk, th = ticks(pid) if pid > 0 else (-1, -1)
        if pid > 0 and tk < 0:
            mine = []
        avail = mem_available_kb()
        prss = sum(max(0, status_kb(p, "VmRSS:")) for p in prof)
        f.write(f"{time.time():.3f}\t{cg_mem()}\t{cg_anon()}\t{cg_cpu_usec()}\t{pid}\t{rss}\t{tk}\t{th}\t{prss}\t{avail}\n")
        if not fired and pid > 0 and 0 <= avail < FLOOR_KB:
            fired = True
            with open(WD, "w") as w:
                w.write(f"{time.time():.3f} MemAvailable {avail // 1024} MiB < floor {FLOOR_KB // 1024} MiB: SIGTERM to the prover "
                        f"(pid {pid}, VmRSS {rss // 1024} MiB), SIGKILL 10 s later\n")
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(10)
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(PERIOD)
