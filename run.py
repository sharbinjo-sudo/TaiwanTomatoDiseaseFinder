import subprocess
import os
import time
import sys
import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== CORRECT PATHS BASED ON YOUR SCREENSHOT =====
DJANGO_PATH = os.path.join(BASE_DIR, "plant_api")

FLUTTER_EXE = os.path.join(
    BASE_DIR,
    "plant_disease",
    "build",
    "windows",
    "x64",
    "runner",
    "Release",
    "plant_disease.exe"   # ✅ this matches your screenshot
)

def run_django():
    print("🚀 Starting Django backend...")
    return subprocess.Popen(
        ["python", "manage.py", "runserver", "127.0.0.1:8000"],
        cwd=DJANGO_PATH,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def run_flutter():
    print("🍅 Launching Flutter Windows app...")
    
    if not os.path.exists(FLUTTER_EXE):
        print("❌ Flutter exe not found at:")
        print(FLUTTER_EXE)
        sys.exit(1)

    return subprocess.Popen(
        [FLUTTER_EXE],
        cwd=os.path.dirname(FLUTTER_EXE),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def main():
    django_process = run_django()
    time.sleep(5)

    flutter_process = run_flutter()

    try:
        flutter_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")

    if django_process.poll() is None:
        print("🔻 Stopping Django server...")
        os.kill(django_process.pid, signal.SIGTERM)

    print("✅ All processes closed.")

if __name__ == "__main__":
    main()
