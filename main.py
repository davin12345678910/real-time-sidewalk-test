import Real_Time_Sidewalk
from ultralytics import RTDETR
from  multiprocessing import Pool, Manager

"""
Definition: this method will allow a user to call the real time sidewalk system 

Parameters:
model - this is the object detection model that we will be using in order to detect objects in a given image 
queue - this is the queue that we will be using for the concurrency of our code 
pool - this is the pool of processes we will be using for the concurrency of our code 

Returns:
None 
"""
def call_real_time_sidewalk(model, queue, pool):
    Real_Time_Sidewalk.init_real_time_sidewalk(model, queue, pool)
    Real_Time_Sidewalk.ocr_stream()


"""
Definition: this method will be used in order to initialize the real time sidewalk system 
"""
if __name__ == '__main__':

    # initialize RTDETR for objects detection of the sidewalk detection model
    model = RTDETR('rtdetr-l.pt')

    # GPU usage to boost runtime
    model = model.to("cuda")

    # This code will be you with having concurrent code
    with Manager() as manager:
        queue = manager.Queue()
        with Pool(processes=10) as pool:
            call_real_time_sidewalk(model, queue, pool)