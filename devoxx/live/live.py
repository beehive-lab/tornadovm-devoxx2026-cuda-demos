#!/usr/bin/env python3
"""A live dashboard for one GPU doing two jobs: a Java LLM engine (jitLLM) writes a GPU kernel in Java, and TornadoVM
runs that Java on the same GPU. Every component lights up while it works, and a GPU panel shows the GPU's load
and which process holds it, in the color of the component that owns that process.

    live.py <workdir>          run by fancyJitllmLive.sh; NO_PAUSE=1 exits at the end without waiting for Enter

Everything shown comes from the processes it starts and from nvidia-smi; nothing is replayed. If the generated kernel
does not compile, the dashboard says so and runs the kernel the same model wrote in rehearsal (reference.kernel).
Standard library only.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_ROOT = os.path.dirname(HERE)
WORK = sys.argv[1]
os.makedirs(WORK, exist_ok=True)

# --- colors ---------------------------------------------------------------------------------------------------
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
MAGENTA, YELLOW, BLUE, ORANGE, CYAN, GREEN, RED, GREY = (
    "\033[1;38;5;207m", "\033[1;38;5;221m", "\033[1;38;5;75m", "\033[1;38;5;208m", "\033[1;38;5;51m",
    "\033[1;38;5;83m", "\033[1;38;5;196m", "\033[38;5;240m")


def visible(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def fit(s, width):
    """Pad or cut a string with color codes to exactly `width` visible columns."""
    out, n, i = [], 0, 0
    while i < len(s) and n < width:
        m = re.match(r"\033\[[0-9;]*m", s[i:])
        if m:
            out.append(m.group(0))
            i += len(m.group(0))
            continue
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out) + RESET + " " * (width - n)


# --- shared state ---------------------------------------------------------------------------------------------
class Component:
    def __init__(self, number, title, subtitle, color):
        self.number, self.title, self.subtitle, self.color = number, title, subtitle, color
        self.state, self.start, self.seconds, self.note = "waiting", None, 0.0, ""

    def activate(self, note=""):
        self.state, self.start, self.note = "active", time.time(), note

    def finish(self, note="", ok=True):
        self.seconds = time.time() - (self.start or time.time())
        self.state, self.note = ("done" if ok else "failed"), note


LLM = Component("1", "jitLLM engine", "Qwen3-4B · Java on the GPU", MAGENTA)
SOURCE = Component("2", "generated Java", "a @Parallel kernel method", YELLOW)
JAVAC = Component("3", "javac", "the kernel + a harness", BLUE)
JIT = Component("4", "TornadoVM JIT", "Java bytecode → CUDA", ORANGE)
KERNEL = Component("5", "the kernel, running", "same GPU, 8K frames", CYAN)
PIPELINE = [LLM, SOURCE, JAVAC, JIT, KERNEL]

lock = threading.Lock()
state = {
    "code": "", "cuda": [], "frame": None, "frame_info": "", "match": None, "tok": "", "view": "code",
    "util": [], "mem": (0, 1), "procs": [], "owners": {}, "message": "", "frames": 0, "frame_ms": [],
    "gpu_name": "", "fallback": False, "seen_gpu": {},
}
done = threading.Event()


# --- GPU monitor (nvidia-smi) ---------------------------------------------------------------------------------
def owner_of(pid):
    """Which of our components started this process: walk its parents up to a process we launched."""
    owners = state["owners"]
    p = pid
    for _ in range(12):
        if p in owners:
            return owners[p]
        try:
            with open(f"/proc/{p}/stat") as f:
                p = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return None
        if p <= 1:
            return None
    return None


def monitor():
    query = subprocess.Popen(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits", "-lms", "250"], stdout=subprocess.PIPE, text=True)
    def apps():
        while not done.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                procs = []
                for line in out.strip().splitlines():
                    pid, mem = [x.strip() for x in line.split(",")]
                    owner = owner_of(int(pid))
                    procs.append((int(pid), int(mem), owner))
                    if owner and not (owner is KERNEL and JIT.state == "active"):  # still compiling: not 'running' yet
                        state["seen_gpu"][owner.title] = (int(pid), max(int(mem), state["seen_gpu"].get(owner.title, (0, 0))[1]))
                with lock:
                    state["procs"] = procs
            except Exception:
                pass
            time.sleep(0.5)
    threading.Thread(target=apps, daemon=True).start()
    for line in query.stdout:
        if done.is_set():
            break
        try:
            name, util, used, total = [x.strip() for x in line.split(",")]
        except ValueError:
            continue
        active = next((c for c in PIPELINE if c.state == "active" and c in (LLM, JIT, KERNEL)), None)
        with lock:
            state["gpu_name"] = name
            state["util"].append((int(util), active.color if active else GREY))  # the whole run, every 250 ms
            state["mem"] = (int(used), int(total))
    query.terminate()


# --- rendering ------------------------------------------------------------------------------------------------
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def box_lines(c, width, tick):
    color = c.color if c.state in ("active", "done") else GREY
    edge = "━" if c.state == "active" else "─"
    corners = ("┏", "┓", "┗", "┛") if c.state == "active" else ("┌", "┐", "└", "┘")
    if c.state == "active":
        status = f"{c.color}{SPIN[tick % 10]} ACTIVE {time.time() - c.start:5.1f} s{RESET}"
    elif c.state == "done":
        status = f"{GREEN}✔ {c.seconds:.1f} s{RESET}" + (f" {DIM}{c.note}{RESET}" if c.note else "")
    elif c.state == "failed":
        status = f"{RED}✘ {c.note}{RESET}"
    else:
        status = f"{GREY}waiting{RESET}"
    title = f"{BOLD if c.state != 'waiting' else ''}{c.number} {c.title}{RESET}"
    inner = width - 2
    return [f"{color}{corners[0]}{edge * inner}{corners[1]}{RESET}",
            f"{color}{'┃' if c.state == 'active' else '│'}{RESET}{fit(' ' + title, inner)}{color}{'┃' if c.state == 'active' else '│'}{RESET}",
            f"{color}{'┃' if c.state == 'active' else '│'}{RESET}{fit(' ' + DIM + c.subtitle, inner)}{color}{'┃' if c.state == 'active' else '│'}{RESET}",
            f"{color}{'┃' if c.state == 'active' else '│'}{RESET}{fit(' ' + status, inner)}{color}{'┃' if c.state == 'active' else '│'}{RESET}",
            f"{color}{corners[2]}{edge * inner}{corners[3]}{RESET}"]


def pipeline_lines(width, tick):
    bw = max(20, (width - 4 * 3) // 5)
    boxes = [box_lines(c, bw, tick) for c in PIPELINE]
    lines = []
    for row in range(5):
        parts = []
        for i, b in enumerate(boxes):
            parts.append(b[row])
            if i < 4:
                flowing = PIPELINE[i].state in ("done",) and PIPELINE[i + 1].state != "waiting"
                parts.append((f"{PIPELINE[i].color} ▶ {RESET}" if flowing else f"{GREY} ▷ {RESET}") if row == 2 else "   ")
        lines.append("".join(parts))
    return lines


def bar(fraction, width, color):
    n = max(0, min(width, round(fraction * width)))
    return f"{color}{'█' * n}{RESET}{GREY}{'░' * (width - n)}{RESET}"


def gpu_lines(width, height):
    with lock:
        util, mem, procs, name = list(state["util"]), state["mem"], list(state["procs"]), state["gpu_name"]
    inner = width - 4
    out = [f"{BOLD}{name or 'GPU'}{RESET}", f"{DIM}one GPU · both workloads{RESET}", ""]
    now = util[-1] if util else (0, GREY)
    out.append(f"utilization {now[0]:3d}% " + bar(now[0] / 100, inner - 17, now[1]))
    out.append(f"memory {mem[0] / 1024:5.1f} GB " + bar(mem[0] / max(mem[1], 1), inner - 17, BLUE))
    out.append("")
    out.append(f"{DIM}utilization, the whole run ({len(util) / 4:.0f} s){RESET}")
    blocks = " ▁▂▃▄▅▆▇█"
    # squeeze the whole run into the panel: each column is the busiest sample of its time slice, in its owner's color
    cols = min(inner, len(util))
    slices = [util[i * len(util) // cols:(i + 1) * len(util) // cols] or util[-1:] for i in range(cols)]
    spark = "".join(f"{c}{blocks[min(8, round(u / 100 * 8))]}{RESET}" for u, c in (max(sl) for sl in slices))
    out.append(spark)
    out.append(f"{MAGENTA}■{RESET} LLM  {ORANGE}■{RESET} JIT  {CYAN}■{RESET} kernel")
    out.append("")
    out.append(f"{BOLD}processes on this GPU{RESET}")
    for pid, used, owner in sorted(procs, key=lambda p: -p[1])[:5]:
        if owner is KERNEL and JIT.state == "active":
            owner = JIT  # the same Java process: TornadoVM is still compiling the kernel to CUDA
        label = f"{owner.color}● {owner.title}{RESET}" if owner else f"{GREY}● other{RESET}"
        out.append(f"{label} {DIM}pid {pid} · {used:,} MiB{RESET}")
    if not procs:
        out.append(f"{GREY}(none right now){RESET}")
    out.append("")
    if state["seen_gpu"]:
        out.append(f"{BOLD}seen on this GPU during the run{RESET}")
    for title, (pid, used) in state["seen_gpu"].items():
        owner = next(c for c in PIPELINE if c.title == title)
        out.append(f"{owner.color}✔ {title}{RESET} {DIM}{used / 1024:.1f} GB{RESET}")
    return [fit(" " + line, width - 2) for line in out][:height]


JAVA_KEYWORDS = set("static void int float for while if break return new".split())


def highlight(line):
    out = []
    for token in re.split(r"(@\w+|\b\d[\d.]*f?\b|\b\w+\b)", line):
        if token.startswith("@"):
            out.append(f"{ORANGE}{token}{RESET}")
        elif re.fullmatch(r"\d[\d.]*f?", token):
            out.append(f"\033[38;5;141m{token}{RESET}")
        elif token in JAVA_KEYWORDS:
            out.append(f"{BLUE}{token}{RESET}")
        elif token in ("IntArray", "FloatArray"):
            out.append(f"\033[38;5;43m{token}{RESET}")
        else:
            out.append(token)
    return "".join(out)


def code_text(raw):
    lines = [l for l in raw.split("\n") if l.strip() not in ("<think>", "</think>", "```java", "```")]
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


# deep blue -> cyan -> white -> gold -> orange -> red -> purple
PALETTE = [17, 18, 19, 20, 21, 27, 33, 39, 45, 51, 87, 123, 159, 195, 230, 229, 228, 227, 221, 215, 209, 203, 197, 161, 125, 89, 53]


def colorizer(rows):
    """Histogram coloring: each escaping pixel gets a color by the rank of its iteration count in this frame, so the
    detail stays visible at any zoom. Pixels that never escaped (inside the set) are black."""
    top = max(max(r) for r in rows)
    values = sorted({v for r in rows for v in r if v < top})
    rank = {v: i / max(1, len(values) - 1) for i, v in enumerate(values)}
    return lambda v: 16 if v >= top else PALETTE[min(len(PALETTE) - 1, int(rank[v] * len(PALETTE)))]


import textwrap

PROMPT = open(os.path.join(HERE, "prompt.txt")).read().strip()
PROMPT_TYPE_SECONDS = 3.0  # the prompt is typed out at the start, while the model loads


def prompt_box(width):
    """The user prompt as a chat bubble, typed out over the first seconds of the run."""
    shown = PROMPT
    if LLM.start is not None:
        typed = int(len(PROMPT) * min(1.0, (time.time() - LLM.start) / PROMPT_TYPE_SECONDS))
        shown = PROMPT[:typed]
    inner = width - 8
    wrapped = textwrap.wrap(shown, inner) or [""]
    if len(shown) < len(PROMPT):
        wrapped[-1] += "▌"
    lines = [f"{MAGENTA}╭─ ❯ the prompt, sent to Qwen3-4B running in jitLLM on the GPU {'─' * max(0, inner - 60)}╮{RESET}"]
    for w in wrapped:
        lines.append(f"{MAGENTA}│{RESET} {BOLD}\033[38;5;231m{fit(w, inner)}{RESET} {MAGENTA}│{RESET}")
    lines.append(f"{MAGENTA}╰{'─' * (inner + 2)}╯{RESET}")
    lines.append(f"  {DIM}+ system prompt: TornadoVM's rules (@Parallel loops, IntArray/FloatArray) and one example kernel{RESET}")
    return lines


def prompt_line(width):
    """One line that keeps the prompt in view after the code panel is gone."""
    return f"{MAGENTA}❯ prompt:{RESET} {DIM}{PROMPT[: width - 16]}…{RESET}"


def assemble_lines(width, height):
    """How the model's answer becomes a program: extract, insert into the harness, javac."""
    a = state.get("assemble") or {}
    steps = a.get("steps", 0)
    tick = lambda n: f"{GREEN}✔{RESET}" if steps > n or (n == 3 and a.get("javac") is not None) else f"{BLUE}▸{RESET}"
    out = [prompt_line(width),
           f"{BLUE}{BOLD}3 javac{RESET}{DIM} · the model's answer becomes a program{RESET}", "",
           f"{tick(1)} {BOLD}extract{RESET}  the ```java block from {a.get('source', '')}: a {a.get('kernel_lines', 0)}-line method",
           f"{tick(2)} {BOLD}insert{RESET}   it at the /*KERNEL*/ marker of Harness.template  ->  Harness.java" if steps >= 2 else "",
           ""]
    if steps >= 2:
        harness = open(os.path.join(WORK, "Harness.java")).read().split("\n")
        k0 = next(i for i, l in enumerate(harness) if "the generated kernel" in l) + 1
        k1 = k0 + a.get("kernel_lines", 0)
        zoom = next((i for i, l in enumerate(harness) if "PHASE zoom" in l), len(harness))  # the check part only
        keep = [i for i, l in enumerate(harness) if i < 3 or "public class Harness" in l or k0 - 1 <= i <= k1
                or i < zoom and any(m in l for m in (".task(\"mandelbrot\"", "plan.execute();", "mandelbrot(w, h, iterations, view, cpu)",
                                        "new TaskGraph(\"check\")", "public static void main"))]
        room = height - len(out) - 6
        shown, previous = [], -1
        for i in keep:
            if i != previous + 1:
                shown.append(f"{GREY}     ┊ …{RESET}")
            line = harness[i]
            if k0 + 3 <= i < k1 - 2:
                # the whole method was just on screen: keep its first and last lines, and the harness around it
                if i == k0 + 3:
                    shown.append(f"{YELLOW}▌{RESET}{YELLOW}     ┊ … {k1 - k0 - 5} more lines written by the model …{RESET}")
                previous = i
                continue
            if k0 <= i < k1:
                shown.append(f"{YELLOW}▌{RESET}{DIM}{i + 1:3d}{RESET} " + highlight(line))
            else:
                note = ""
                if ".task(" in line:
                    note = f"  {ORANGE}◀ on the GPU{RESET}"
                elif "mandelbrot(w, h, iterations, view, cpu)" in line:
                    note = f"  {GREEN}◀ same method, CPU{RESET}"
                shown.append(f" {GREY}{i + 1:3d} {line}{RESET}{note}")
            previous = i
        out += shown[:room]
        out.append(f"{DIM}  {YELLOW}▌{RESET}{DIM} = written by the model; grey = the fixed harness around it{RESET}")
        out.append("")
    if steps >= 3:
        out.append(f"{tick(3)} {BOLD}compile{RESET}  javac -cp $TORNADOVM_HOME/share/java/tornado/*.jar Harness.java")
        if a.get("errors"):
            out += [f"  {RED}{l}{RESET}" for l in a["errors"].strip().split("\n")[:6]]
        else:
            out.append(f"    {GREEN}-> Harness.class, {a.get('class_bytes', 0):,} bytes of bytecode, {a.get('javac', 0):.1f} s{RESET}"
                       f"{DIM}  · next: TornadoVM JIT to CUDA{RESET}")
    return out


def content_lines(width, height, tick):
    with lock:
        view, code, cuda, frame, info = state["view"], state["code"], list(state["cuda"]), state["frame"], state["frame_info"]
    out = []
    if view == "code":
        out += prompt_box(width - 4)
        out.append("")
        out.append(f"{YELLOW}{BOLD}2 generated Java{RESET}{DIM} · the model's answer, streaming as it writes it{RESET}")
        lines = code_text(code)
        if not lines and LLM.state == "active":
            out.append(f"{MAGENTA}{SPIN[tick % 10]}{RESET} jitLLM is loading Qwen3-4B (7.5 GB) onto the GPU and JIT-compiling its own")
            out.append(f"  Java inference kernels with TornadoVM; then it reads the prompt and starts writing.")
            out.append(f"  {DIM}(watch the GPU memory climb on the left){RESET}")
        cursor = f"{YELLOW}▌{RESET}" if LLM.state == "active" and tick % 6 < 3 else ""
        room = max(4, height - len(out) - 1)
        for i, l in enumerate(lines[-room:]):
            last = i == len(lines[-room:]) - 1
            out.append(f"{DIM}{len(lines) - len(lines[-room:]) + i + 1:3d} │{RESET} " + highlight(l) + (cursor if last else ""))
    elif view == "assemble":
        out += assemble_lines(width, height)
    elif view == "cuda":
        out.append(prompt_line(width))
        out.append(f"{ORANGE}{BOLD}4 TornadoVM JIT{RESET}{DIM} · the CUDA it generated from the model's Java, for this GPU{RESET}")
        out.append("")
        for l in cuda[: height - 4]:
            out.append(f"\033[38;5;180m{l}{RESET}")
    else:
        out.append(prompt_line(width))
        out.append(f"{CYAN}{BOLD}5 the generated kernel, running on the GPU{RESET}{DIM} · {info}{RESET}")
        if frame:
            rows = frame[: 2 * (height - 3)]
            color = colorizer(rows)
            cols = min(width - 2, len(rows[0]))
            for top_row, bottom_row in zip(rows[0::2], rows[1::2]):
                cells = []
                for c in range(cols):
                    i = c * len(top_row) // cols
                    # one cell, two pixels: the upper half block in the top pixel's color, the background the bottom's
                    cells.append(f"\033[38;5;{color(top_row[i])};48;5;{color(bottom_row[i])}m▀")
                out.append("".join(cells) + RESET)
    return [fit(" " + l, width - 2) for l in out] + [" " * (width - 2)] * max(0, height - len(out))


def footer():
    with lock:
        parts = []
        if state["tok"]:
            parts.append(f"{MAGENTA}jitLLM {state['tok']}{RESET}")
        if state["match"]:
            same, total = state["match"]
            parts.append(f"{ORANGE}GPU = CPU on {100 * same / total:.2f}% of {total:,} pixels{RESET}")
        if state["frame_ms"]:
            ms = sorted(state["frame_ms"])
            parts.append(f"{CYAN}{state['frames']} frames · {ms[len(ms) // 2]:.1f} ms per 8K frame{RESET}")
        if state["fallback"]:
            parts.append(f"{RED}live kernel did not compile: rehearsal kernel{RESET}")
        if state["message"]:
            parts.append(state["message"])
    return "  " + "   ".join(parts)


def render(tick):
    cols, rows = shutil.get_terminal_size((160, 48))
    width = max(120, cols)
    gw = 46
    cw = width - gw - 1
    body_h = max(20, rows - 10)
    lines = [f"{BOLD} jitLLM writes a GPU kernel · TornadoVM runs it on the same GPU{RESET}"
             f"{DIM}   — a Java LLM engine and the Java it generates, one RTX 4090{RESET}"]
    lines += pipeline_lines(width, tick)
    g = gpu_lines(gw, body_h - 2)
    c = content_lines(cw, body_h - 2, tick)
    lines.append(f"{GREY}┌{'─' * (gw - 2)}┐{RESET} {GREY}┌{'─' * (cw - 2)}┐{RESET}")
    for i in range(body_h - 2):
        left = g[i] if i < len(g) else " " * (gw - 2)
        lines.append(f"{GREY}│{RESET}{left}{GREY}│{RESET} {GREY}│{RESET}{c[i]}{GREY}│{RESET}")
    lines.append(f"{GREY}└{'─' * (gw - 2)}┘{RESET} {GREY}└{'─' * (cw - 2)}┘{RESET}")
    lines.append(fit(footer(), width))
    sys.stdout.write("\033[H" + "\n".join(fit(l, width) for l in lines[: rows - 1]) + "\033[J")
    sys.stdout.flush()


def renderer():
    tick = 0
    while not done.is_set():
        render(tick)
        tick += 1
        time.sleep(0.1)
    render(tick)


# --- the steps ------------------------------------------------------------------------------------------------
def shell(command, **kwargs):
    """A bash command with the demo environment (env.sh), in its own process."""
    return subprocess.Popen(["bash", "-c", f'source "{DEMO_ROOT}/env.sh"; {command}'], **kwargs)


def generate():
    LLM.activate()
    prompt = open(os.path.join(HERE, "prompt.txt")).read().strip()
    err = open(os.path.join(WORK, "generation.err"), "w")
    p = shell(f'use_jitllm; cd "$JITLLM_DIR"; exec ./jitllm run --gpu --cuda-graphs --model "$MODEL_CODE" -c 2048 '
              f'--max-new-tokens 600 --temperature 0 -sp "$(cat "{HERE}/system.txt")" --prompt "$(cat "{HERE}/prompt.txt") /no_think"',
              stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
    state["owners"][p.pid] = LLM
    raw = []
    while True:
        ch = p.stdout.read(1)
        if not ch:
            break
        raw.append(ch)
        with lock:
            state["code"] = "".join(raw)
        if SOURCE.state == "waiting" and "```java" in state["code"]:
            SOURCE.activate()
    p.wait()
    err.close()
    text = "".join(raw)
    open(os.path.join(WORK, "generation.out"), "w").write(text)
    m = re.search(r"achieved tok/s: ([0-9.]+)\. Tokens: (\d+), seconds: ([0-9.]+)", open(os.path.join(WORK, "generation.err")).read())
    with lock:
        state["tok"] = f"{float(m.group(1)):.0f} tok/s, {m.group(2)} tokens (prompt + answer) in {float(m.group(3)):.1f} s" if m else "no metrics"
    LLM.finish("written", ok=p.returncode == 0)
    kernel = re.search(r"```java\n(.*?)```", text, re.S)
    SOURCE.finish(f"{len(kernel.group(1).splitlines())} lines" if kernel else "", ok=bool(kernel))
    return kernel.group(1) if kernel else ""


def build(kernel, source="the model's answer"):
    """The model's method -> Harness.java (at the /*KERNEL*/ marker of Harness.template) -> javac. Each step is
    recorded for the 'assemble' view."""
    with lock:
        state["view"] = "assemble"
        state["assemble"] = {"source": source, "kernel_lines": len(kernel.strip().splitlines()), "steps": 1,
                             "javac": None, "class_bytes": 0, "errors": ""}
    time.sleep(1.0)
    body = kernel.rstrip().replace("\n", "\n    ")
    with open(os.path.join(WORK, "Harness.java"), "w") as f:
        f.write(open(os.path.join(HERE, "Harness.template")).read().replace("/*KERNEL*/", body))
    with lock:
        state["assemble"]["steps"] = 2
    time.sleep(1.5)
    t = time.time()
    p = shell(f'use_tornadovm_7; cd "{WORK}"; javac -cp "$TORNADO_CP" Harness.java',
              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    open(os.path.join(WORK, "javac.log"), "w").write(out)
    cls = os.path.join(WORK, "Harness.class")
    with lock:
        state["assemble"].update(steps=3, javac=time.time() - t, errors=out if p.returncode else "",
                                 class_bytes=os.path.getsize(cls) if p.returncode == 0 and os.path.exists(cls) else 0)
    time.sleep(float(os.environ.get("ASSEMBLE_SECONDS", "4")))  # time to read it on stage
    return p.returncode == 0


def run_kernel():
    p = shell(f'use_tornadovm_7; cd "{WORK}"; exec tornado --printKernel --classpath . Harness 100 wait',
              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    state["owners"][p.pid] = KERNEL  # re-labelled below while it JIT-compiles
    JIT.activate()
    log = open(os.path.join(WORK, "run.log"), "w")
    cuda, in_cuda, rows, frame_meta = [], False, [], None
    cuda_shown = None
    for line in p.stdout:
        log.write(line)
        line = line.rstrip("\n")
        if line.startswith('extern "C" __global__') and cuda_shown is None:
            # the first kernel compiled (the check); the zoom's task graph prints its own copy later, not shown
            in_cuda = True
            with lock:
                state["view"] = "cuda"
            cuda_shown = time.time()
        if in_cuda and len(cuda) < 60:
            cuda.append(line[:200])
            with lock:
                state["cuda"] = list(cuda)
        if line.startswith("MATCH"):
            in_cuda = False
            _, same, total = line.split()
            with lock:
                state["match"] = (int(same), int(total))
            JIT.finish("Java → CUDA")
            # let the audience read the CUDA, then start the zoom
            time.sleep(max(0.0, 4.0 - (time.time() - (cuda_shown or time.time()))))
            KERNEL.activate()
            with lock:
                state["view"] = "frame"
            p.stdin.write("go\n")
            p.stdin.flush()
        elif line.startswith("FRAME"):
            _, number, ms, scale = line.split()
            frame_meta, rows = (int(number), float(ms), float(scale)), []
        elif line.startswith("ROW") and frame_meta:
            rows.append(list(map(int, line.split()[1:])))
            if len(rows) == 72:
                with lock:
                    state["frame"] = rows
                    state["frames"] = frame_meta[0] + 1
                    state["frame_ms"].append(frame_meta[1])
                    state["frame_info"] = f"frame {frame_meta[0] + 1} · {frame_meta[1]:.1f} ms on the GPU · zoom ×{2.6 / frame_meta[2]:,.0f}"
        elif line.startswith("END"):
            KERNEL.finish(f"{state['frames']} frames")
    p.wait()
    log.close()
    if KERNEL.state == "active":
        KERNEL.finish("stopped", ok=False)
    if JIT.state == "active":
        JIT.finish("failed", ok=False)
    return p.returncode == 0


def main():
    sys.stdout.write("\033[?1049h\033[?25l")  # alternate screen, hide the cursor
    threading.Thread(target=monitor, daemon=True).start()
    painter = threading.Thread(target=renderer, daemon=True)
    painter.start()
    ok = False
    try:
        kernel = generate()
        JAVAC.activate()
        compiled = bool(kernel) and build(kernel)
        if not compiled:
            JAVAC.finish("live kernel failed", ok=False)
            with lock:
                state["fallback"] = True
            time.sleep(2)
            JAVAC.activate("rehearsal kernel")
            compiled = build(open(os.path.join(HERE, "reference.kernel")).read(), "reference.kernel (rehearsal)")
        JAVAC.finish("compiled" if compiled else "failed", ok=compiled)
        JAVAC.seconds = (state.get("assemble") or {}).get("javac") or JAVAC.seconds  # javac itself, not the reading pauses
        if compiled:
            ok = run_kernel()
        match = state["match"]
        ok = ok and match is not None and match[0] >= 0.99 * match[1] and KERNEL.state == "done"
        with lock:
            state["message"] = (f"{GREEN}{BOLD}PASSED{RESET}" if ok else f"{RED}{BOLD}FAILED{RESET}") + \
                (f"{DIM}  [enter] to exit{RESET}" if not os.environ.get("NO_PAUSE") else "")
        time.sleep(1.5)  # one more GPU sample for the panel
        if not os.environ.get("NO_PAUSE"):
            input()
    finally:
        done.set()
        painter.join(timeout=1)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    # a plain summary on the normal screen, for the record
    seen = ", ".join(f"{t} (pid {pid}, {mem:,} MiB)" for t, (pid, mem) in state["seen_gpu"].items())
    print(f"  jitLLM: {state['tok']}")
    print(f"  kernel: {state['frames']} frames, median {sorted(state['frame_ms'])[len(state['frame_ms']) // 2] if state['frame_ms'] else 0:.1f} ms per 8K frame"
          + (f"; GPU = CPU on {100 * state['match'][0] / state['match'][1]:.2f}% of pixels" if state["match"] else ""))
    print(f"  on the GPU: {seen or 'not sampled'}" + ("; live kernel failed to compile, rehearsal kernel used" if state["fallback"] else ""))
    print(f"  logs: {WORK}")
    print("LiveDashboard: PASSED -- the Java LLM wrote the kernel and it ran on the same GPU" if ok else "LiveDashboard: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
