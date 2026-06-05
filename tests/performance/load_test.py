import time
import concurrent.futures
import requests
import os
import json
import numpy as np

import sys
import os

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration
API_URL = "http://localhost:8000"
TEST_FILE = os.path.join(PROJECT_ROOT, "test_case_34.npy")
CONCURRENT_USERS = [1, 5, 10, 20] # Scaling load
REQUISITIONS_PER_USER = 2

def simulate_user(user_id):
    results = []
    for i in range(REQUISITIONS_PER_USER):
        start = time.time()
        try:
            with open(TEST_FILE, 'rb') as f:
                files = {'file': (os.path.basename(TEST_FILE), f, 'application/octet-stream')}
                response = requests.post(f"{API_URL}/analyze", files=files, timeout=30)
                
            latency = (time.time() - start) * 1000 # ms
            results.append({
                "user_id": user_id,
                "request_id": i,
                "status": response.status_code,
                "latency_ms": latency
            })
        except Exception as e:
            results.append({
                "user_id": user_id,
                "request_id": i,
                "status": "error",
                "error": str(e)
            })
    return results

def run_load_test():
    print("\n" + "="*60)
    print("ATRIONNET: LOAD BALANCE & SCALABILITY STRESS TEST")
    print("="*60)
    
    # Check if backend is up
    try:
        requests.get(f"{API_URL}/health", timeout=5)
    except:
        print("ERROR: Backend not running at http://localhost:8000. Start it first!")
        return

    all_stats = {}
    
    for num_users in CONCURRENT_USERS:
        print(f"[CLINICAL-VAL] Simulating {num_users} concurrent users...")
        
        start_batch = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            future_to_user = {executor.submit(simulate_user, i): i for i in range(num_users)}
            
            batch_latencies = []
            errors = 0
            for future in concurrent.futures.as_completed(future_to_user):
                user_results = future.result()
                for res in user_results:
                    if res['status'] == 200:
                        batch_latencies.append(res['latency_ms'])
                    else:
                        errors += 1
                        
        end_batch = time.time()
        
        avg_lat = np.mean(batch_latencies) if batch_latencies else 0
        p95_lat = np.percentile(batch_latencies, 95) if batch_latencies else 0
        throughput = (num_users * REQUISITIONS_PER_USER) / (end_batch - start_batch)
        
        print(f"   -> Avg Latency: {avg_lat:.2f} ms")
        print(f"   -> P95 Latency: {p95_lat:.2f} ms")
        print(f"   -> Throughput : {throughput:.2f} req/sec")
        print(f"   -> Error Rate : {errors / (num_users * REQUISITIONS_PER_USER) * 100:.1f}%")
        
        all_stats[num_users] = {
            "avg_latency": avg_lat,
            "p95_latency": p95_lat,
            "throughput": throughput,
            "errors": errors
        }

    # Archive results
    results_path = "f:/Final_Year/Final_Semester_one/Final_Year_Research_Project/AtrionNet_Implementation/tests/reports/load_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
        
    # Generate High-Res Plot for Thesis
    from tests.reporting.plot_utils import save_scalability_plot
    plot_path = "f:/Final_Year/Final_Semester_one/Final_Year_Research_Project/AtrionNet_Implementation/tests/reports/scalability_plot.png"
    save_scalability_plot(all_stats, plot_path)
        
    print("="*60)
    print(f"[CLINICAL-VAL] Scalability telemetry & plot archived at {results_path}\n")

if __name__ == "__main__":
    run_load_test()
