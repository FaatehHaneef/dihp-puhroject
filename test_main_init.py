"""Test main.py initialization"""
import sys
import time
import threading

def run_main():
    try:
        from main import main
        print("Starting main()...")
        main()
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Run in thread with timeout
thread = threading.Thread(target=run_main, daemon=True)
thread.start()
thread.join(timeout=8)  # 8 second timeout

if thread.is_alive():
    print("\nMain is running (initialization succeeded)")
else:
    print("\nMain exited")
