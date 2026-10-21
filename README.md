Install the Dependencies 
python -m pip install mujoco gymnasium stable-baselines3 numpy imageio imageio-ffmpeg packaging

Verify Dependencies
python -c "import mujoco, gymnasium, stable_baselines3, numpy, imageio; print('All dependencies installed successfully')"

cd ~/Turtle_MuJoCo
source ~/mujoco_venv/bin/activate

python test_env.py

python train.py

python evaluate.py
