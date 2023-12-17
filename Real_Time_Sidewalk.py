# these are the imports that wee will need for our system to run
import cv2
import replicate
import os
import re
from PIL import Image

# GLOBAL CONSTANTS

# this is the object detection model that we will be using
RETDETR_model = None

# this is where we will be storing all of the results from the tasks in which we will be using
task_queue = None

# this is the pool of processes we will be using for our tasks
task_pool = None

# this is the text that we will be outputting to the user
text = []

"""
Definition: this is where we will be initalizing the needed variables in our code 

Variables:
model - this is the model that we will be using for object detection 
queue - this is where we will be storing the output of our tasks 
pool - this is where we will be getting the processes in which we can 
       use to process each of the tasks in which we will want to use 

Returns: None
"""


def init_real_time_sidewalk(model, queue, pool):
    global RETDETR_model
    RETDETR_model = model

    global task_queue
    task_queue = queue

    global task_pool
    task_pool = pool


"""
Definition: this method will get us text and caption information from llava 

Parameters:
path - this is the path of the image that we want to analyze
queue - this is the queue where we will be storing all of the information that we want 
task - this is the type of task that we want to do, whether it be caption or text 

Return: None
"""


def get_llava(path, queue, task):
    # this is the replicate token that we will need to use
    os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

    # Give me a caption for the image. If it includes text, please include.
    if task == "text":

        # this is the api call we will be using for text
        output = replicate.run(
            "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
            input={
                "image": open(path,
                              "rb"),
                "prompt": """Question: What number is in the image? Answer: """}
        )
        # The yorickvp/llava-13b model can stream output as it's running.
        # The predict method returns an iterator, and you can iterate over that output.
        text = ""
        for item in output:
            text += item

        queue.put(["text", text])
    else:

        # this is the api call that we will be using for captions
        output = replicate.run(
            "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
            input={
                "image": open(path,
                              "rb"),
                "prompt": """Give me a caption for the image"""}
        )
        # The yorickvp/llava-13b model can stream output as it's running.
        # The predict method returns an iterator, and you can iterate over that output.
        text = ""
        for item in output:
            text += item

        queue.put(["caption", text])


#################### main program code #################################################################################################


"""
Definition: this is where we will be running the main portion of our code 
that will allow users to get information from nearby pedestrian singals

Parameters: None

Returns: None 
"""


def ocr_stream():
    import time
    captures = 0  # Number of still image captures during view session

    cap = cv2.VideoCapture(
        'C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\bandicam 2023-12-15 16-13-05-414.mp4')  # Starts reading the video stream in dedicated thread

    # this is what we will be using if you want to get webcam footage
    # cap = cv2.VideoCapture(0)

    # Main display loop
    print("\nPUSH c TO GET CROSSWALK INFORMATION (press s TO CLEAR MESSAGE). PRESS Q TO QUIT THE SYSTEM\n")

    # here we will be starting up the system
    while True:

        # Here we will be getting user input and seeing whether the user
        # wants information or wants to quit
        pressed_key = cv2.waitKey(1)

        # here is where we will be getting a frame from the user
        ret, frame = cap.read()

        # this is where we will be storing the current frame
        # in which we will be working with
        global current_frame
        current_frame = frame

        frame = current_frame

        # Resize the cropped image back to original size
        cv2.imwrite("C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\current.png", current_frame)

        # if the user presses s, we will erase the text from the screen
        global text
        if pressed_key == ord("s"):
            global text
            text = []

        start_time = 0
        # this is where we will be capturing data to output to the user
        if pressed_key == ord('c'):
            start_time = time.time()
            # last_execution_time = current_time
            global RETDETR_model
            results = RETDETR_model(['C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\current.png'])

            '''''''''
                Here we will be processing all of the yolo information which we will 
                later use in order to get the bounding boxes for each object which will
                be used to crop the image 
                '''
            boxes = None
            classes = None
            for key in results:
                boxes = key.boxes.xyxy.tolist()
                classes = key.boxes.cls.tolist()

            current_image = Image.open(
                'C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\current.png')
            cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\cropped_imgs\\" + "cropped_im" + str(
                1) + ".png"
            current_image.save(cropped_image_path)

            index = 2

            global task_queue
            global task_pool

            task = []

            # here we will be looking through all of the detected objects and
            # will be checking which are traffic signals
            for i in range(len(boxes)):

                # 9 is the class for a traffic signal
                if classes[i] == 9:

                    box = boxes[i]

                    # here we will cropping the image and placing the image in the directory
                    image = Image.open('C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\current.png')
                    crop_area = (
                        int(round(float(box[0]), 2)), int(round(float(box[1]), 2)), int(round(float(box[2]), 2)),
                        int(round(float(box[3]), 2)))
                    cropped_image = image.crop(crop_area)
                    cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\cropped_imgs\\" + "cropped_im" + str(
                        index) + ".png"
                    cropped_image.save(cropped_image_path)

                    # this is the threshold that we will use to determine if something is a pedestrian
                    # traffic signal or not
                    threshold = 0.85
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    aspect_ratio = width / height

                    # Check if it's a square, because from the given traffic signals you can detect
                    # traffic signals that can be seen are in a square shape.
                    # Note: if there are traffic signals that are edge cases please let me know, but
                    # so far it is working quite well
                    if aspect_ratio >= threshold:
                        task.append((cropped_image_path, task_queue, "text"))
                        task.append((cropped_image_path, task_queue, "caption"))
                        break

                    index = index + 1

            # here we will be running all of the task through the use of the multiprocessing method
            task_pool.starmap(get_llava, task)

            # Here we will be getting all of the text that we have
            # and we will customize it for the user
            time = ""
            caption = ""

            while not task_queue.empty():
                current = task_queue.get()
                if current[0] == "text":

                    print("Time: ", current[1])
                    # Regular expression to find numbers
                    numbers = re.findall(r'\d+', current[1])

                    # Convert extracted strings to integers
                    numbers = [int(num) for num in numbers]

                    if (len(numbers) == 0):
                        time = "No time given"
                    else:
                        time = str(numbers[0])

                else:
                    caption = current[1]

            text = ["Caption: " + caption, "Time remaining: " + time]

        # here we will be printing out the total time for debugging purposes
        import time
        end_time = time.time()

        if pressed_key == ord("c"):
            print("ELAPSED TIME: ", (end_time - start_time))

        # Here we will be outputing the text to the user
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)  # Green color in BGR
        thickness = 1  # Minimum thickness
        lineType = cv2.LINE_AA

        # Display each line of text
        y_position = 50
        for line in text:
            cv2.putText(frame, line.strip(), (10, y_position), font, fontScale, color, thickness, lineType)
            y_position += 40  # Adjust line spacing as needed

        # here we will be displaying the text to the user. For our final system
        # we will be reading outloud the given information
        cv2.imshow("realtime OCR", frame)

        # This is the code that we will run if the user decides to end the method
        if pressed_key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("OCR stream stopped\n")
            print("{} image(s) captured and saved to current directory".format(captures))
            break