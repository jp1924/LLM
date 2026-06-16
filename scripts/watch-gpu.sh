watch -n 3 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used --format=csv,noheader && echo && nvidia-smi pmon -c 1'
