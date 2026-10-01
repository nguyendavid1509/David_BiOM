import time
import os
import psutil
import joblib
import numpy as np

MODEL_PATH = "linear_regression.pkl"
N_TESTS = 20
N_INFERENCES = 10000

process = psutil.Process(os.getpid())

# -------------------------
# Model loading benchmark
# -------------------------

start = time.perf_counter()

model = joblib.load(MODEL_PATH)

load_time = time.perf_counter() - start

print(f"Model loading time: {load_time * 1000:.3f} ms")


# ---------------------------------------------------------------------------
# RAM benchmark
# ---------------------------------------------------------------------------

ram_after_load = process.memory_info().rss

print(f"RAM usage after loading model: {ram_after_load / 1024**2:.2f} MB")


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

process.cpu_percent(interval=None)

start = time.perf_counter()

for _ in range(N_TESTS):
    prediction = model.predict(np.random.rand(N_INFERENCES).reshape(-1,1))

inference_time = time.perf_counter() - start

cpu = process.cpu_percent(interval=None)

print(f"Total inference time: {inference_time:.4f} ms")
print(f"Average inference time: {inference_time / N_TESTS * 1000:.4f} ms")
print(f"CPU usage: {cpu:.1f}%")
print(f"Peak RAM usage: {}")

