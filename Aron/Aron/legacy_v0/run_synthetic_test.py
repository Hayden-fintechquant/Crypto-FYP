"""執行 tests/test_backtest_engine.py 裡的所有測試
顯示結果"""

import subprocess
import sys

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test/test_backtest_engine.py", "-v"],
        capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr)
    sys.exit(result.returncode)