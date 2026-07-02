.\scripts\run_eval.ps1                 # asks for model name → normal checkpoint, deterministic
.\scripts\run_eval.ps1 -Stochastic     # sample actions instead of argmax
.\scripts\run_eval.ps1 -Best           # use best_checkpoint.pth
.\scripts\run_eval.ps1 -Best -Stochastic
It prompts Model (run) name, checks models/<run_name>/ exists, then loads checkpoint.pth (or best_checkpoint.pth with -Best) from there. Two decision flags exactly as you asked — -Stochastic and -Best — plus an optional -Episodes (default 10) and -RunName if you'd rather skip the prompt. The richer per-wrapper --inspect_* diagnostics still live in eval.py itself if you ever want them.