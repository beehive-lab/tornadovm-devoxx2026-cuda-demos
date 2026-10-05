# Condenses TornadoVM's console profiler JSON to one line per task (used by demoHybrid.sh).
/^iteration|^All iterations/ { print; next }
/"METHOD"/           { m = $2; gsub(/[",]/, "", m) }
/"BACKEND"/          { b = $2; gsub(/[",]/, "", b) }
/"TASK_KERNEL_TIME"/ { t = $2; gsub(/[",]/, "", t); printf "      %-26s %-5s kernel %6.1f us\n", m, b, t / 1000 }
