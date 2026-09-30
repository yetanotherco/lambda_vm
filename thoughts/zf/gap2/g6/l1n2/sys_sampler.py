#!/usr/bin/env python3
"""Host-wide sampler for the L1N2-noise job (runs ON THE BOX for the whole job, standard library only).

Every --interval seconds (default 0.25) it writes one S line of host-wide numbers, and every --procs-every samples one
P line per container process that used CPU since the last P line. Only numbers and each process's own short name
(/proc/<pid>/comm, kept only if it is made of [A-Za-z0-9_:./-], else `?`) are written: no command line, no environment.

  sys_sampler.py --out <tsv> [--interval 0.25] [--procs-every 4] [--root /]
  sys_sampler.py --selftest [--root /]

Output lines (tab-separated):
  # sys_sampler v1 · interval <s> · cpus <n> · keys <S field names>
  S <t> <field values in the header's order>        (-1: unreadable)
  P <t> <pid> <comm> <cpu jiffies since the last P line>

S fields: mhz_min mhz_mean mhz_max (scaling_cur_freq over all CPUs) · tctl tccd1 tccd2 (k10temp, °C) · busy total
(/proc/stat jiffies, all CPUs) · psi_cpu_some psi_mem_some psi_mem_full psi_io_some psi_io_full (µs totals) · load1 ·
memfree memavail cached dirty writeback anonhuge shmem (kB) · pgmajfault pgfault compact_stall pgscan_direct
pgscan_kswapd thp_fault_alloc (vmstat counts) · cg_nr_throttled cg_throttled_ns · cg_usage_ns.
Stops on SIGTERM or SIGINT.
"""
import argparse
import os
import re
import signal
import sys
import time

STOP = False
NAME_OK = re.compile(r"^[A-Za-z0-9_:./-]{1,32}$")
MEMINFO = ("MemFree", "MemAvailable", "Cached", "Dirty", "Writeback", "AnonHugePages", "Shmem")
VMSTAT = ("pgmajfault", "pgfault", "compact_stall", "pgscan_direct", "pgscan_kswapd", "thp_fault_alloc")
FIELDS = (["mhz_min", "mhz_mean", "mhz_max", "tctl", "tccd1", "tccd2", "busy", "total",
           "psi_cpu_some", "psi_mem_some", "psi_mem_full", "psi_io_some", "psi_io_full", "load1"]
          + [m.lower() for m in MEMINFO] + list(VMSTAT) + ["cg_nr_throttled", "cg_throttled_ns", "cg_usage_ns"])


def _stop(*_):
    global STOP
    STOP = True


def read(root, rel):
    try:
        with open(os.path.join(root, rel.lstrip("/"))) as f:
            return f.read()
    except OSError:
        return None


class Host:
    def __init__(self, root):
        self.root = root
        cpu_dir = os.path.join(root, "sys/devices/system/cpu")
        self.freq = sorted(p for p in (os.path.join(cpu_dir, d, "cpufreq/scaling_cur_freq")
                                       for d in (os.listdir(cpu_dir) if os.path.isdir(cpu_dir) else [])
                                       if re.fullmatch(r"cpu[0-9]+", d)) if os.path.exists(p))
        self.temps = {"tctl": None, "tccd1": None, "tccd2": None}
        hw = os.path.join(root, "sys/class/hwmon")
        for h in sorted(os.listdir(hw)) if os.path.isdir(hw) else []:
            if (read(root, f"sys/class/hwmon/{h}/name") or "").strip() != "k10temp":
                continue
            for f in sorted(os.listdir(os.path.join(hw, h))):
                m = re.fullmatch(r"temp([0-9]+)_label", f)
                if not m:
                    continue
                label = (read(root, f"sys/class/hwmon/{h}/{f}") or "").strip().lower()
                if label in self.temps:
                    self.temps[label] = f"sys/class/hwmon/{h}/temp{m.group(1)}_input"
        self.cg = next((p for p in ("sys/fs/cgroup/cpu/cpu.stat", "sys/fs/cgroup/cpu.stat")
                        if read(root, p) is not None), None)
        self.cg_usage = next((p for p in ("sys/fs/cgroup/cpu/cpuacct.usage", "sys/fs/cgroup/cpuacct/cpuacct.usage")
                              if read(root, p) is not None), None)

    def sample(self):
        out = []
        mhz = []
        for p in self.freq:
            try:
                with open(p) as f:
                    mhz.append(int(f.read()) / 1000.0)
            except (OSError, ValueError):
                pass
        out += [min(mhz), sum(mhz) / len(mhz), max(mhz)] if mhz else [-1, -1, -1]
        for key in ("tctl", "tccd1", "tccd2"):
            v = read(self.root, self.temps[key]) if self.temps[key] else None
            out.append(int(v) / 1000.0 if v and v.strip().lstrip("-").isdigit() else -1)
        stat = read(self.root, "proc/stat") or ""
        m = re.search(r"^cpu +(.*)$", stat, re.M)
        if m:
            v = [int(x) for x in m.group(1).split()]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            out += [sum(v) - idle, sum(v)]
        else:
            out += [-1, -1]
        for res, kinds in (("cpu", ("some",)), ("memory", ("some", "full")), ("io", ("some", "full"))):
            txt = read(self.root, f"proc/pressure/{res}") or ""
            for kind in kinds:
                m = re.search(rf"^{kind} .*total=([0-9]+)", txt, re.M)
                out.append(int(m.group(1)) if m else -1)
        la = (read(self.root, "proc/loadavg") or "").split()
        out.append(float(la[0]) if la else -1)
        mem = read(self.root, "proc/meminfo") or ""
        for key in MEMINFO:
            m = re.search(rf"^{key}: +([0-9]+)", mem, re.M)
            out.append(int(m.group(1)) if m else -1)
        vm = read(self.root, "proc/vmstat") or ""
        for key in VMSTAT:
            m = re.search(rf"^{key} ([0-9]+)", vm, re.M)
            out.append(int(m.group(1)) if m else -1)
        cg = read(self.root, self.cg) if self.cg else None
        for key in ("nr_throttled", "throttled_time"):
            m = re.search(rf"^{key} ([0-9]+)", cg or "", re.M)
            if not m and key == "throttled_time":
                m2 = re.search(r"^throttled_usec ([0-9]+)", cg or "", re.M)
                out.append(int(m2.group(1)) * 1000 if m2 else -1)
            else:
                out.append(int(m.group(1)) if m else -1)
        u = read(self.root, self.cg_usage) if self.cg_usage else None
        out.append(int(u) if u and u.strip().isdigit() else -1)
        return out

    def procs(self):
        """{pid: (comm, utime + stime jiffies)} for every process /proc lists."""
        res = {}
        proc = os.path.join(self.root, "proc")
        for d in os.listdir(proc):
            if not d.isdigit():
                continue
            st = read(self.root, f"proc/{d}/stat")
            if not st:
                continue
            r = st.rfind(")")
            fields = st[r + 2:].split()
            try:
                cpu = int(fields[11]) + int(fields[12])
            except (IndexError, ValueError):
                continue
            comm = (read(self.root, f"proc/{d}/comm") or "?").strip()
            res[int(d)] = (comm if NAME_OK.match(comm) else "?", cpu)
        return res


def run(out, interval, procs_every, root):
    host = Host(root)
    cpus = len(host.freq)
    last = host.procs()
    n = 0
    with open(out, "w") as f:
        f.write(f"# sys_sampler v1 · interval {interval} · cpus {cpus} · keys {' '.join(FIELDS)}\n")
        nxt = time.time()
        while not STOP:
            t = time.time()
            f.write("S\t%.4f\t%s\n" % (t, "\t".join(str(x) for x in host.sample())))
            n += 1
            if n % procs_every == 0:
                now = host.procs()
                for pid, (comm, cpu) in now.items():
                    d = cpu - last.get(pid, (comm, cpu))[1] if pid in last else cpu
                    if d > 0:
                        f.write("P\t%.4f\t%d\t%s\t%d\n" % (t, pid, comm, d))
                last = now
                f.flush()
            nxt += interval
            time.sleep(max(0.0, nxt - time.time()))


def selftest(root):
    host = Host(root)
    s = host.sample()
    assert len(s) == len(FIELDS), (len(s), len(FIELDS))
    named = dict(zip(FIELDS, s))
    p = host.procs()
    assert p, "no processes read"
    print("SELFTEST OK · cpus %d · mhz %s · tctl %s · busy/total %s/%s · psi_cpu_some %s · memavail %s kB · "
          "cg_throttled_ns %s · %d processes" % (len(host.freq), named["mhz_mean"], named["tctl"], named["busy"],
                                                 named["total"], named["psi_cpu_some"], named["memavailable"],
                                                 named["cg_throttled_ns"], len(p)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--procs-every", type=int, default=4)
    ap.add_argument("--root", default="/")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.root)
        sys.exit(0)
    if not a.out:
        ap.error("--out is required")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    run(a.out, a.interval, a.procs_every, a.root)
