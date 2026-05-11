import sys
import time

# Timeout after 5 seconds to avoid hanging on camera init
import signal
def timeout_handler(signum, frame):
    print("Timeout - initialization took too long")
    sys.exit(0)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)

try:
    from main import main
    print("Starting main()...")
    main()
except KeyboardInterrupt:
    print("Interrupted")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    signal.alarm(0)
