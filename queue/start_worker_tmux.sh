#!/usr/bin/env bash
set -euo pipefail

SESSION="spines_queue"
PROJECT_DIR="$HOME/code/SpinesGUI"
QUEUE_DIR="$PROJECT_DIR/queue"
DB_PATH="$QUEUE_DIR/jobs.sqlite"
CONDA_ENV="spinesGUI"

mkdir -p "$QUEUE_DIR" "$QUEUE_DIR/logs"

LAUNCH_LOG="$QUEUE_DIR/tmux_launch.log"
echo "[$(date)] start_worker_tmux.sh called" >> "$LAUNCH_LOG"

# conda init
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# already running?
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[$(date)] Worker already running in tmux session: $SESSION" | tee -a "$LAUNCH_LOG"
  exit 0
fi

CMD="source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && cd $PROJECT_DIR && python -u worker.py --db $DB_PATH --poll 2 |& tee -a $QUEUE_DIR/worker_stdout.log"

echo "[$(date)] Starting tmux session '$SESSION' with command:" >> "$LAUNCH_LOG"
echo "$CMD" >> "$LAUNCH_LOG"

tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""

echo "[$(date)] Started worker in tmux session: $SESSION" | tee -a "$LAUNCH_LOG"
echo "[$(date)] DB: $DB_PATH" | tee -a "$LAUNCH_LOG"
