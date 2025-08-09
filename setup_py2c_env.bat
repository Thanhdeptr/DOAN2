@echo off
REM === Đường dẫn tới Python 3.10 ===
set PYTHON310_PATH=D:\DOAN2\pythontoC\python.exe

REM === Tên môi trường ảo ===
set ENV_NAME=py2c_env

REM === Di chuyển tới thư mục hiện tại ===
cd /d %~dp0

echo [+] Tạo môi trường ảo bằng Python 3.10 ...
%PYTHON310_PATH% -m venv %ENV_NAME%

echo [+] Kích hoạt môi trường ...
call %ENV_NAME%\Scripts\activate.bat

echo [+] Cài đặt pip và các thư viện cần thiết ...
python -m pip install --upgrade pip
pip install tensorflow==2.10 keras==2.10 numpy

echo [✓] Đã cài xong môi trường và thư viện cho Py2C.
pause
