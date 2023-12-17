# real-time-sidewalk

# installation
- you will need to get your own video to put into videocapture line 115 of real_time_sidewalk.py
- You will also need to get the checkpoint for rtdetr-l.pt
- if you want to get a test video, you can use bandicam to record a test video 


## steps to use the system 
1. `git clone https://github.com/davin12345678910/real-time-sidewalk.git`
2. `conda create --name real-time-sidewalk python=3.8`
3. `conda activate real-timep-sidewalk`
4. install torch `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
   (Note: for the above step you might need to get cudatoolkit installed, so make sure that is installed)
5. `pip install ultralytics`
6. `pip install multiprocessing`
7. `pip install replicate`
8. `pip install opencv-python`
9. `pip install pillow`


## how to run code
run `python main.py`

### Note: if you want to just test llava.py with our prompts run:
- `python llava_test.py`