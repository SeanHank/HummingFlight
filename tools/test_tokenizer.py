"""Quick test for the apply_chat command of tokenizer_server.py"""
import json
import subprocess
import sys

python_exe = sys.executable
script = r"C:\Users\Administrator\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a6ef2cb6183a09d4ab8082a\tools\tokenizer_server.py"
model_dir = r"D:\glm-5.2-bf16"

proc = subprocess.Popen(
    [python_exe, script, model_dir],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8"
)

# Read the ready signal
ready = proc.stdout.readline()
print("READY:", ready.strip())

# Test apply_chat
req = json.dumps({"cmd": "apply_chat", "messages": [{"role": "user", "content": "hello"}], "add_generation_prompt": True})
proc.stdin.write(req + "\n")
proc.stdin.flush()
resp = proc.stdout.readline()
print("RESP:", resp.strip())

# Test encode
req2 = json.dumps({"cmd": "encode", "text": "hello world", "add_special": False})
proc.stdin.write(req2 + "\n")
proc.stdin.flush()
resp2 = proc.stdout.readline()
print("ENCODE:", resp2.strip())

# Quit
proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
proc.stdin.flush()
proc.wait()
