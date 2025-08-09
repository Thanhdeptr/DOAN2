@echo off
echo Installing Vision Mamba 2 dependencies...

echo.
echo Installing PyTorch and torchvision...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing mamba-ssm...
pip install mamba-ssm

echo.
echo Installing other dependencies...
pip install numpy matplotlib tqdm tensorboard scikit-learn seaborn

echo.
echo Testing installation...
python -c "from mamba_ssm import Mamba; print('Mamba SSM installed successfully!')"

echo.
echo Installation completed!
echo You can now run: python train.py
pause 