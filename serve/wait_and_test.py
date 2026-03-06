"""Wait for Ray Serve to be ready then test endpoints."""
import time
import json
import urllib.request
import ray
from ray import serve

ray.init(address="ray://ray-head:10001", ignore_reinit_error=True)

# Wait for serve RUNNING
for i in range(1, 31):
    try:
        s = serve.status()
        for name, app in s.applications.items():
            status = str(app.status)
            print(f"  Attempt {i}: {name} = {status}")
            if status == "ApplicationStatus.RUNNING":
                print("Serve is RUNNING! ✅")
                break
        else:
            time.sleep(3)
            continue
        break
    except Exception as e:
        print(f"  Attempt {i}: {e}")
        time.sleep(3)
else:
    print("Timed out waiting for serve")
    exit(1)

time.sleep(2)

# Test endpoints
print("\n=== Testing endpoints ===")
resp = urllib.request.urlopen("http://ray-head:8000/health").read()
print(f"GET /health: {resp.decode()}")

data = json.dumps({"messages": [{"role": "user", "content": "Hello Ray Serve!"}]}).encode()
req = urllib.request.Request(
    "http://ray-head:8000/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req).read()
result = json.loads(resp)
print(f'POST /v1/chat/completions: {result["choices"][0]["message"]["content"]}')
print("All tests passed! ✅")
