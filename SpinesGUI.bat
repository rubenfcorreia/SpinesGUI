@echo off

powershell -Command "ssh ddream 'bash -lc ""~/code/SpinesGUI/queue/start_worker_tmux.sh && source ~/miniconda3/etc/profile.d/conda.sh && conda activate spinesGUI && python ~/code/SpinesGUI/SpinesGUI.py""'"

pause
