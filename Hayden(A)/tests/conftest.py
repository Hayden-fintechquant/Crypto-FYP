import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest, yaml
from src.synth import make_synthetic


@pytest.fixture(scope="session")
def cfg():
    c = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config_fyp1.yaml"), encoding="utf-8"))
    c["sample"]["start"], c["sample"]["end"] = "2019-07-01", "2024-12-23"
    c["screening"]["min_avg_volume_usd"] = 0      # 合成資料的成交量不是美元尺度，不套用真實的 US$1M 門檻
    return c


@pytest.fixture(scope="session")
def synth():
    return make_synthetic()
