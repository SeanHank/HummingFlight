"""GLM-5.2 Tokenizer subprocess service

Invoked by the C++ main process via stdin/stdout pipes, line-delimited JSON protocol.
Ensures Tokenizer semantics are 100% consistent with the official transformers implementation.

Usage (launched by C++):
    python tokenizer_server.py <model_dir>

Protocol (one JSON per line):
    In:  {"cmd":"encode","text":"hello","add_special":false}
    Out: {"ids":[109377,11]}
    In:  {"cmd":"decode","ids":[109377]}
    Out: {"text":"hello"}
    In:  {"cmd":"apply_chat","messages":[{"role":"user","content":"hello"}]}
    Out: {"ids":[151331,...]}
    In:  {"cmd":"info"}
    Out: {"vocab_size":154880,"eos":154820,"bos":null,...}
    In:  {"cmd":"quit"}
    Out: (none, exits)
"""

import json
import os
import sys

# Single source of truth: scripts/sync_version.py (do not edit by hand).
__version__ = "2026.9.2"

# Suppress stderr warnings emitted while importing transformers/torch (e.g. "PyTorch was not found").
# These warnings would leak into the stdout pipe and interfere with the ready-signal detection on the C++ side.
_devnull = open(os.devnull, "w")
_old_stderr = sys.stderr
sys.stderr = _devnull
try:
    from transformers import AutoTokenizer
finally:
    sys.stderr = _old_stderr


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: tokenizer_server.py <model_dir>\n")
        sys.exit(1)

    model_dir = sys.argv[1]
    # trust_remote_code=True: required by the GLM family
    tk = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Cache basic tokenizer info
    info = {
        "vocab_size": tk.vocab_size,
        "bos": tk.bos_token_id,
        "eos": tk.eos_token_id,
        "pad": tk.pad_token_id,
        "has_chat_template": tk.chat_template is not None,
    }

    # Ready signal (the C++ side starts sending commands after reading this line)
    sys.stdout.write(json.dumps({"ready": True, "info": info}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"error": "bad json: " + str(e)}) + "\n")
            sys.stdout.flush()
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "quit":
                break
            elif cmd == "info":
                resp = info
            elif cmd == "encode":
                text = req.get("text", "")
                add_special = req.get("add_special", True)
                ids = tk.encode(text, add_special_tokens=add_special)
                resp = {"ids": ids}
            elif cmd == "decode":
                ids = req.get("ids", [])
                skip_special = req.get("skip_special", True)
                text = tk.decode(ids, skip_special_tokens=skip_special)
                resp = {"text": text}
            elif cmd == "apply_chat":
                messages = req.get("messages", [])
                add_gen = req.get("add_generation_prompt", True)
                text = tk.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_gen
                )
                ids = tk.encode(text, add_special_tokens=False)
                resp = {"ids": ids, "text": text}
            else:
                resp = {"error": "unknown cmd: " + str(cmd)}
        except Exception as e:
            resp = {"error": str(e)}

        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
