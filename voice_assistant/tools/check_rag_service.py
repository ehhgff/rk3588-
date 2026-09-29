#!/usr/bin/env python3
import socket
import json
import sys

try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect('/tmp/rag_optimized.sock')
    sock.send(json.dumps({'action': 'ping'}).encode())
    response = sock.recv(1024).decode()
    sock.close()
    data = json.loads(response)
    if data.get('status') == 'ok':
        sys.exit(0)
except:
    pass
sys.exit(1)
