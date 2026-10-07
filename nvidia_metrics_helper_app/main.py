import subprocess, time
from prometheus_client import start_http_server, Gauge

gpu_util = Gauge("nvidia_gpu_utilization_percent", "GPU compute utilization")
mem_used = Gauge("nvidia_gpu_memory_used_bytes", "VRAM used")
mem_total = Gauge("nvidia_gpu_memory_total", "VRAM total")
power_draw = Gauge("nvidia_gpu_power_draw", "Power draw")
power_limit = Gauge("nvidia_gpu_power_limit", "Power limit")

def collect():
    # calls nvidia-smi, and returns results to gpu_util, mem_used, and power_draw
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ], text=True)
    util, used, total, p_draw, p_limit = (x.strip() for x in out.splitlines()[0].split(","))
    gpu_util.set(float(util))
    mem_used.set(float(used))
    mem_total.set(float(total))
    power_draw.set(float(p_draw))
    power_limit.set(float(p_limit))

if __name__ == '__main__':
    start_http_server(8091)

    while True:
        collect()
        time.sleep(5)
