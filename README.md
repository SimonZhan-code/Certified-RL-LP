# Set up
```python
conda create -n jointdiff python=3.7
conda activate jointdiff
pip install pip==21.0.1
pip install -r requirements.txt
pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```


# Run the examples, run each individual py files such as 
```python
python PJ4.py
python Ball4.py
python pendulum.py
python LK.py
python D6.py
python AttControl.py
```

Control Animation for the learned controller from our approach. Dashed lines are the target position.
![alt text](AttControl.gif "Control Animation for the learned controller from our approach. Dashed lines are the target position.")