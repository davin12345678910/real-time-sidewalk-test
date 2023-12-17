import Real_Time_Sidewalk
from ultralytics import RTDETR
from  multiprocessing import Pool, Manager

def main(model, queue, pool):

    # This is where OCR is started...
    Real_Time_Sidewalk.init_real_time_sidewalk(model, queue, pool)
    Real_Time_Sidewalk.ocr_stream()


if __name__ == '__main__':
    # To run in IDE (instead of commamnd line), comment out main() and uncomment the block below:

    # this is the model that we will be using in order to detect
    # the objects that are given in an image
    model = RTDETR('rtdetr-l.pt')

    # this is to initalize cuda in order to use a GPU
    model = model.to("cuda")

    # here we will be passing the queue and the pool in which we will
    # be using to have tasks be processed in parallel
    with Manager() as manager:
        queue = manager.Queue()
        with Pool(processes=10) as pool:
            main(model, queue, pool)