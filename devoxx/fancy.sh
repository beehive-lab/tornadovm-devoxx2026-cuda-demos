# Shared helpers for the fancy*.sh demos: a live spinner around each step, and the renderer (fancy.py).
# Sourced after env.sh; nothing to run by hand.
# shellcheck shell=bash

FANCY="python3 $DEMO_ROOT/fancy.py"
FANCY_LOGS="${FANCY_LOGS:-$(mktemp -d)}"
export FANCY_STATE="$FANCY_LOGS/scoreboard.tsv"
: > "$FANCY_STATE"

# spin LABEL LOG COMMAND...: runs the command with its output in LOG, animating one status line until it ends
spin() {
    local label="$1" log="$2"; shift 2
    local frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' start=$SECONDS i=0
    ( "$@" ) > "$log" 2>&1 &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        printf '\r    \033[1;36m%s\033[0m %-44s \033[2m%3ds\033[0m ' "${frames:i++%10:1}" "$label" $((SECONDS - start))
        sleep 0.1
    done
    local status=0
    wait "$pid" || status=$?
    if [ "$status" -eq 0 ]; then
        printf '\r    \033[1;32m✔\033[0m %-44s \033[2m%3ds\033[0m \n' "$label" $((SECONDS - start))
    else
        printf '\r    \033[1;31m✘\033[0m %-44s \033[2m%3ds  (log: %s)\033[0m\n' "$label" $((SECONDS - start)) "$log"
    fi
    return "$status"
}

# fancy_pause: [enter] between acts unless NO_PAUSE is set
fancy_pause() { [ -n "${NO_PAUSE:-}" ] || read -r -p $'\033[2m    [enter] next \033[0m' _; }
