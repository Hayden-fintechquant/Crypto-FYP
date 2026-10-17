"""預先登記鎖: 把 config.yaml 的雜湊值存起來，之後正式執行時檢查有沒有被改過。"""
import hashlib, json, datetime
import yaml


def config_hash(cfg_path) -> str:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    canon = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def lock(cfg_path, lock_path):
    h = config_hash(cfg_path)
    info = dict(sha256=h, locked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def verify(cfg_path, lock_path):
    try:
        with open(lock_path, encoding="utf-8") as f:
            info = json.load(f)
    except FileNotFoundError:
        return False, "找不到 prereg.lock，請先執行: python run_all.py --lock", None
    h = config_hash(cfg_path)
    if h != info["sha256"]:
        return False, "config.yaml 在鎖定之後被修改過! 不能做正式執行。", info
    return True, "預先登記檢查通過", info
