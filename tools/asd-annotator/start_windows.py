"""Start the separate local annotation program using the existing WSL runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

SERVICE = 'scenerecall-asd-annotator'
ASSET = 'asset_666d87eb4e274e1fa9b7'
PORT = 8766


def existing(state_file: Path):
    try:
        value = json.loads(state_file.read_text(encoding='utf-8'))
        if value.get('service') != SERVICE or value.get('asset_id') != ASSET or value.get('port') != PORT:
            return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{PORT}/api/health', timeout=1) as response:
            health = json.load(response)
        if (health.get('service') == SERVICE and health.get('asset_id') == ASSET
                and health.get('pid') == value.get('pid') and health.get('ready') is True):
            token = value.get('token')
            if isinstance(token, str) and token and all(char.isalnum() or char in '-_' for char in token):
                return f'http://127.0.0.1:{PORT}/?token={token}'
    except (OSError, ValueError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description='启动独立的辉夜姬人工标注小程序')
    parser.add_argument('--no-open', action='store_true', help='启动后台服务，不自动打开浏览器')
    args = parser.parse_args()
    if os.name != 'nt':
        raise SystemExit('Windows 启动器请在 Windows 运行；WSL/Linux 请使用 start.sh。')
    folder = Path(__file__).resolve().parent
    repo = folder.parent.parent
    data = repo / 'training/lrasd/manual-kaguya'
    data.mkdir(parents=True, exist_ok=True)
    state_file = data / 'server.private.json'
    url = existing(state_file)
    launched = None
    if url is None:
        flags = subprocess.CREATE_NO_WINDOW
        converted = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'wslpath', '-u', (folder / 'start.sh').as_posix()],
                                   capture_output=True, text=True, check=True, creationflags=flags, timeout=30)
        script = converted.stdout.strip()
        if not script.startswith('/') or '\n' in script:
            raise SystemExit('无法定位 WSL 启动脚本。')
        with (data / 'server.private.log').open('ab') as output, (data / 'server-errors.log').open('ab') as errors:
            launched = subprocess.Popen(['wsl.exe', '-d', 'Ubuntu', '--', 'bash', script],
                                        stdin=subprocess.DEVNULL, stdout=output, stderr=errors, creationflags=flags)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            url = existing(state_file)
            if url:
                break
            if launched.poll() is not None:
                raise SystemExit(f'启动失败，请查看 {data / "server-errors.log"}')
            time.sleep(.4)
    if not url:
        raise SystemExit(f'服务尚未就绪，请查看 {data / "server-errors.log"}。SceneRecall 主应用不受影响。')
    print(f'独立标注程序已就绪：http://127.0.0.1:{PORT}/')
    print(f'标注保存在：{data}')
    if not args.no_open:
        os.startfile(url)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f'启动失败：{error}。请确认 WSL Ubuntu 与现有 SceneRecall 运行环境可用。')
