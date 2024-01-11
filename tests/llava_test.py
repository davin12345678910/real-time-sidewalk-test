import replicate
import os
import cv2

os.environ["REPLICATE_API_TOKEN"] = "r8_HfRAhxwo6UJWnpdEPkLhpwC9HIRxGzn0fHb38"

file_path = "C:\\Users\\davin\\PycharmProjects\\real-time-sidewalk\\cropped_imgs\\cropped_im6.png"

output = replicate.run(
    "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
    input={"image": open(file_path, "rb"), "prompt" : """Question: What number is in the image? Answer: """}
)
# The yorickvp/llava-13b model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
for item in output:
    # https://replicate.com/yorickvp/llava-13b/api#output-schema
    print(item, end="")


output2 = replicate.run(
    "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
    input={"image": open("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\text-detection\\example#13.png", "rb"), "prompt" : """Give me a caption for the image"""}
)
# The yorickvp/llava-13b model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
for item in output2:
    # https://replicate.com/yorickvp/llava-13b/api#output-schema
    print(item, end="")