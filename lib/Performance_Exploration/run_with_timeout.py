#!/usr/bin/env python3

import subprocess
import sys
import time
import psutil

def kill_process_tree(pid):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()

def run_command_with_timeout(command, timeout):
    proc = subprocess.Popen(command, shell=True)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc.pid)
        sys.exit("Command timed out")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python run_with_timeout.py <timeout> <command>")
    timeout = int(sys.argv[1])
    command = " ".join(sys.argv[2:])
    run_command_with_timeout(command, timeout)
