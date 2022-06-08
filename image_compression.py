import cv2 as cv
import os
import numpy as np
import time


PATH_INPUT = r"Z:\testy_kompresja\input"
PATH_OUTPUT = r"Z:\testy_kompresja\output"

path_tmp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"temp.jpg")

DEBUG = False
JPEG_COMPRESSION_QUALITY = 50
PNG_SAVING_COMPRESSION = 3 # 0-fast -> 9-good compression
RESIZE_SCALE = 4

# init vars
size_all_initial = 0
size_all_compressed = 0
list_compression_times = list()

def compress(img):
    w,h = img.shape[1],img.shape[0]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img,[w//RESIZE_SCALE,h//RESIZE_SCALE], interpolation=cv.INTER_NEAREST)
    # save file with compression as jpeg
    cv.imwrite(
        path_tmp_file, 
        img, 
        [cv.IMWRITE_JPEG_QUALITY, JPEG_COMPRESSION_QUALITY, 
        cv.IMWRITE_JPEG_OPTIMIZE, 1,
        cv.IMWRITE_JPEG_PROGRESSIVE, 1], 
    )
    img = cv.imread(path_tmp_file, 0) # flag 0 - read as grayscale
    # remove temp jpeg file
    os.remove(path_tmp_file)
    #img = cv.resize(img,[w,h], interpolation=cv.INTER_NEAREST)
    return img


for i,file in enumerate(os.listdir(PATH_INPUT)):
    # file paths
    path_in = os.path.join(PATH_INPUT, file)
    path_out = os.path.join(PATH_OUTPUT, file)

    time_start = time.time()
    
    img = cv.imread(path_in)   
    # change from BGR to GRAYSCALE 
    img = compress(img)
    # save as png
    cv.imwrite(path_out, img, [cv.IMWRITE_PNG_COMPRESSION, PNG_SAVING_COMPRESSION]) 
    # calc time
    compression_time = time.time()-time_start
    # calc  file sizes
    initial_size = os.stat(path_in).st_size
    compressed_size = os.stat(path_out).st_size
    # print file info
    print(f'''{i+1}/{len(os.listdir(PATH_INPUT))} 
    File {file} size: {initial_size/1e3:.2f} KB -> {compressed_size/1e3:.2f} KB [{compressed_size/initial_size*100:.2f}%]
    Compressed in {compression_time} s
    ''')

    size_all_initial += initial_size
    size_all_compressed += compressed_size
    list_compression_times.append(compression_time)

    if DEBUG:
        cv.destroyAllWindows()
        h = 1080//2
        w = img.shape[1]//(img.shape[0]//h)

        img_debug = np.zeros((h,w), img.dtype)
        img_debug = np.vstack([cv.resize(cv.cvtColor(cv.imread(path_in), cv.COLOR_BGR2GRAY) , [w,h]),cv.resize(img, [w,h])])

        cv.imshow("Debug", img_debug)
        cv.waitKey(25)

print(f'''
Processed {len(os.listdir(PATH_INPUT))} files. 
Size: {size_all_initial/1e6:.2f} MB -> {size_all_compressed/1e6:.2f} MB [{size_all_compressed/size_all_initial*100:.2f}%]
Mean time: {np.mean(np.array(list_compression_times))}
''')
