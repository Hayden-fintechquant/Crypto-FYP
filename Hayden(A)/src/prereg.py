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
        return False, f"找不到 {lock_path}，請先鎖定設定檔 (--lock)", None
    h = config_hash(cfg_path)
    if h != info["sha256"]:
        return False, f"{cfg_path} 在鎖定之後被修改過! 不能做正式執行 (要改請重新 --lock 並在報告說明原因)。", info
    return True, "預先登記檢查通過", info
