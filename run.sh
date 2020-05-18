#!/bin/sh

srun -pcpu --cpus-per-task=4 julia --project --optimize --math-mode=fast --check-bounds=no PINCAT.jl $1