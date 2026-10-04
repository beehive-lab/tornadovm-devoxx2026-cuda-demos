#!/usr/bin/env python3
"""Terminal rendering for demo 28 (standard library only; NO_COLOR=1 for plain text).

    render.py stream          stdin -> terminal as tokens arrive, without the model's empty <think></think> or ``` fences
    render.py code FILE       a Java source with line numbers and syntax colors
    render.py cuda LOG [N]    the first N lines of the CUDA kernel TornadoVM generated (from tornado.printKernel)
    render.py fractal LOG     the kernel's result as an image, GPU vs CPU time, and how many pixels match
"""
import os
import re
import sys

COLOR = os.environ.get("NO_COLOR") is None
WIDTH = 46
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED = "\033[1;32m", "\033[1;33m", "\033[1;31m"


def paint(color, s):
    return f"{color}{s}{RESET}" if COLOR else s


def visible(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def pad(s, width):
    return s + " " * max(0, width - visible(s))


def read(path):
    with open(path, errors="replace") as f:
        return re.sub(r"\033\[[0-9;]*m", "", f.read())


def bars(title, rows, unit, fmt="{:,.1f}"):
    """rows: (label, value, color), on one scale; the smallest value gets a star."""
    top = max(v for _, v, _ in rows)
    best = min(v for _, v, _ in rows)
    print("    " + paint(BOLD, title))
    for label, value, color in rows:
        n = max(1, round(value / top * WIDTH))
        star = paint(GREEN, " ★") if value == best else ""
        print(f"    {pad(label, 34)} {paint(color, '█' * n)}{' ' * (WIDTH - n)} {fmt.format(value):>9} {unit}{star}")
    print()


JAVA_KEYWORDS = set("""abstract boolean break byte case catch char class continue default do double else extends final
    finally float for if implements import int long new package private protected public return short static super
    switch this throw throws try void while var true false null""".split())


def highlight_java(line):
    """Keywords, annotations, numbers, strings and comments in color."""
    if line.strip().startswith("//"):
        return paint(DIM, line)
    out = []
    for token in re.split(r"(@\w+|\"[^\"]*\"|\b\d[\d_.]*[fFL]?\b|\b\w+\b)", line):
        if token.startswith("@"):
            out.append(paint("\033[1;38;5;208m", token))
        elif token.startswith('"'):
            out.append(paint(GREEN, token))
        elif re.fullmatch(r"\d[\d_.]*[fFL]?", token):
            out.append(paint("\033[0;38;5;141m", token))
        elif token in JAVA_KEYWORDS:
            out.append(paint("\033[1;38;5;75m", token))
        elif token in ("IntArray", "FloatArray", "Math"):
            out.append(paint("\033[0;38;5;43m", token))
        else:
            out.append(token)
    return "".join(out)


def stream():
    """Copy stdin to the terminal as it arrives, indented, without the model's empty <think></think> or the fences."""
    hidden = {"<think>", "</think>", "```java", "```"}
    pending, at_line_start, started = "", True, False
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            break
        pending += ch
        # hold back text that may be the start of a hidden marker, until it is decided
        if any(h.startswith(pending.strip()) for h in hidden) and pending.strip() and ch != "\n":
            continue
        if pending.strip() in hidden:
            pending = ""
            continue
        for c in pending:
            if not started and c in "\n ":
                continue  # nothing shown yet: skip the blank lines the hidden markers leave
            started = True
            if at_line_start and c != "\n":
                sys.stdout.write("    " + paint(DIM, "│ "))
                at_line_start = False
            sys.stdout.write(paint("\033[0;38;5;152m", c) if c != "\n" else c)
            at_line_start = c == "\n"
        sys.stdout.flush()
        pending = ""
    print()


def code(path):
    lines = open(path).read().rstrip("\n").split("\n")
    for n, line in enumerate(lines, 1):
        print(f"    {paint(DIM, f'{n:3d} │')} {highlight_java(line)}")
    print()


def cuda(log, count="16"):
    text = read(log)
    start = text.find('extern "C" __global__')
    if start < 0:
        print(paint(DIM, "    (no CUDA source in the log)"))
        return
    lines = text[start:].split("\n")[: int(count)]
    print("    " + paint(BOLD, "...which TornadoVM JIT-compiled to CUDA") + paint(DIM, f"  (first {count} lines, --printKernel)"))
    for line in lines:
        line = line if len(line) <= 108 else line[:105] + "..."
        print("    " + paint(DIM, "┊ ") + paint("\033[0;38;5;109m", line))
    print()


def fractal(log):
    text = read(log)
    gpu = re.search(r"GPU_MS ([0-9.]+)", text)
    cpu = re.search(r"CPU_MS ([0-9.]+)", text)
    match = re.search(r"MATCH (\d+) (\d+)", text)
    rows = [list(map(int, line.split()[1:])) for line in text.splitlines() if line.startswith("ROW ")]
    if not (gpu and cpu and match and rows):
        print(paint(RED, "    the kernel did not run (see the log)"))
        return
    top = max(max(r) for r in rows) or 1
    palette = [17, 18, 19, 20, 26, 32, 38, 44, 50, 86, 122, 158, 194, 229, 223, 217, 211, 205, 199, 163]
    for row in rows:
        cells = []
        for v in row:
            if v >= top:
                cells.append("\033[48;5;16m " if COLOR else "#")
            else:
                level = palette[min(len(palette) - 1, int((v / top) ** 0.5 * len(palette)))]
                cells.append(f"\033[48;5;{level}m " if COLOR else " .:-=+*%@"[min(8, v * 9 // top)])
        print("    " + "".join(cells) + (RESET if COLOR else ""))
    print()
    g, c = float(gpu.group(1)), float(cpu.group(1))
    same, total = int(match.group(1)), int(match.group(2))
    bars("4096 × 3072 pixels, up to 256 iterations each",
         [("same method, Java, 1 CPU thread", c, YELLOW), ("same method, GPU via TornadoVM", g, GREEN)], "ms", fmt="{:,.1f}")
    print(f"    {paint(BOLD, f'{100 * same / total:.2f}%')} of {total:,} pixels identical to the CPU run"
          + paint(DIM, "  (the rest: float rounding at the set's edge; the GPU fuses multiply-adds)"))
    print()


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    {"stream": stream, "code": code, "cuda": cuda, "fractal": fractal}[command](*args)
