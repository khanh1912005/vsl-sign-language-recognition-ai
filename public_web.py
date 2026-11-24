# -*- coding: utf-8 -*-
"""
KHỞI CHẠY & PUBLIC WEB VSL AI LÊN INTERNET (HTTPS + WEBSOCKET + WEBCAM)
=======================================================================
Tập lệnh này tự động:
1. Khởi chạy máy chủ FastAPI (`web_server.py`) tại cổng 8000.
2. Mở đường hầm HTTPS công khai (Cloudflare Quick Tunnel) không cần tài khoản.
3. In ra đường link `https://xxxx.trycloudflare.com` để bất kỳ ai trên điện thoại
   hoặc máy tính khác đều có thể truy cập cùng lúc, bật Webcam (nhờ chuẩn HTTPS)
   và sử dụng mô hình TFLite.
"""
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUDFLARED_PATH = os.path.join(BASE_DIR, "cloudflared.exe")
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-windows-amd64.exe"
)


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ensure_cloudflared() -> str:
    if os.path.exists(CLOUDFLARED_PATH) and os.path.getsize(CLOUDFLARED_PATH) > 10_000_000:
        return CLOUDFLARED_PATH
    print("[*] Đang tải công cụ Cloudflare HTTPS Tunnel (cloudflared.exe)...")
    urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
    print("[OK] Đã tải xong cloudflared.exe!")
    return CLOUDFLARED_PATH


def is_port_in_use(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.chdir(BASE_DIR)
    cf_bin = ensure_cloudflared()

    server_proc = None
    if not is_port_in_use(8000):
        print("[*] Đang khởi động máy chủ AI (web_server.py) trên cổng 8000...")
        server_proc = subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "web_server.py")],
            cwd=BASE_DIR,
        )
        # Chờ cổng 8000 sẵn sàng
        for _ in range(40):
            if is_port_in_use(8000):
                break
            time.sleep(0.5)
    else:
        print("[OK] Máy chủ web_server.py đã đang chạy sẵn trên cổng 8000.")

    print("[*] Đang tạo đường link HTTPS công khai trên Internet...")
    tunnel_proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")

    def wait_until_link_is_live(url: str, timeout_sec: int = 90) -> bool:
        """
        Link trycloudflare.com mới tạo cần ~10-60 giây để bản ghi DNS lan truyền.
        Nếu mở trình duyệt quá sớm sẽ gặp lỗi DNS_PROBE_POSSIBLE và Windows sẽ
        ghi nhớ (cache) kết quả lỗi đó. Vì vậy ta hỏi thẳng DNS 1.1.1.1 (không
        đụng vào cache của máy) cho tới khi tên miền phân giải được, sau đó xóa
        cache DNS của Windows rồi mới thông báo link cho người dùng.
        """
        host = url.replace("https://", "").strip("/")
        t_start = time.time()
        while time.time() - t_start < timeout_sec:
            try:
                out = subprocess.run(
                    ["nslookup", host, "1.1.1.1"],
                    capture_output=True, text=True, timeout=10,
                    encoding="utf-8", errors="replace",
                ).stdout
                if host in out and ("104." in out or "172." in out or "2606:" in out):
                    break
            except Exception:
                pass
            waited = int(time.time() - t_start)
            print(f"\r[*] Đang chờ Cloudflare kích hoạt tên miền... {waited}s", end="", flush=True)
            time.sleep(3)
        else:
            print()
            return False
        subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
        # Kiểm tra thực tế link đã trả về trang web chưa
        for _ in range(10):
            try:
                with urllib.request.urlopen(url + "/api/stats", timeout=10) as resp:
                    if resp.status == 200:
                        print()
                        return True
            except Exception:
                time.sleep(2)
        print()
        return True

    def read_tunnel_output():
        nonlocal public_url
        for line in tunnel_proc.stdout:
            match = url_pattern.search(line)
            if match and not public_url:
                public_url = match.group(0)
                lan_ip = get_lan_ip()
                print(f"[*] Cloudflare đã cấp tên miền: {public_url}")
                live = wait_until_link_is_live(public_url)
                print("\n" + "=" * 74)
                if live:
                    print("🌐 ĐÃ PUBLIC WEB LÊN MẠNG THÀNH CÔNG! (HỖ TRỢ HTTPS + WEBCAM + ĐA NGƯỜI DÙNG)")
                else:
                    print("⚠️  Link đã tạo nhưng DNS chưa kích hoạt xong — đợi thêm 1 phút rồi mở lại.")
                print("=" * 74)
                print(f"👉 Link truy cập toàn cầu (Internet HTTPS) : {public_url}")
                print(f"👉 Link mạng nội bộ (Wi-Fi/LAN)            : http://{lan_ip}:8000")
                print(f"👉 Link trên máy hiện tại (Localhost)       : http://localhost:8000")
                print("-" * 74)
                print("💡 Bạn có thể gửi đường link https://...trycloudflare.com ở trên cho bất kỳ")
                print("   ai (trên điện thoại, laptop khác mạng) để truy cập cùng lúc!")
                print("   (Nhấn Ctrl + C tại cửa sổ này khi muốn đóng đường link công khai)")
                print("=" * 74 + "\n", flush=True)
                try:
                    with open(os.path.join(BASE_DIR, "PUBLIC_URL.txt"), "w", encoding="utf-8") as f:
                        f.write(f"PUBLIC_HTTPS_URL={public_url}\nLAN_URL=http://{lan_ip}:8000\nLOCAL_URL=http://localhost:8000\n")
                except Exception:
                    pass

    t = threading.Thread(target=read_tunnel_output, daemon=True)
    t.start()

    try:
        while tunnel_proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Đang dừng đường hầm Public Web...")
    finally:
        tunnel_proc.terminate()
        if server_proc is not None:
            server_proc.terminate()


if __name__ == "__main__":
    main()
