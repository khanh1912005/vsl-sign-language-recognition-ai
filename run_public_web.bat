@echo off
chcp 65001 >nul
title VSL AI - Public Web Len Mang (HTTPS + Multi-User)
cd /d "%~dp0"
python public_web.py
pause
