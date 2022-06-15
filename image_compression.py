import cv2 as cv
import os
import numpy as np
import time

# PARAMETERS
PATH_INPUT = r"Z:\testy_kompresja\input"
PATH_OUTPUT = r"Z:\testy_kompresja\output"

JPEG_COMPRESSION_QUALITY = 50
PNG_SAVING_COMPRESSION = 3 # 0-fast -> 9-good compression
RESIZE_SCALE = 4

def compress(img, path_tmp_file=""):
    if path_tmp_file == "":  path_tmp_file=  os.path.join(os.path.dirname(os.path.abspath(__file__)),"temp.jpg")
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
    img = cv.resize(img,[w,h], interpolation=cv.INTER_NEAREST)
    return img

def compress_images_in_dir(pth_input, pth_output, debug = False):
    path_tmp_file =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"temp.jpg")
    if debug:
        # INIT VARS
        size_all_initial = 0
        size_all_compressed = 0
        list_compression_times = list()
    for i,file in enumerate(os.listdir(pth_input)):
        # file paths
        path_in = os.path.join(pth_input, file)
        path_out = os.path.join(pth_output, file)

        if debug: time_start = time.time()
        
        img = cv.imread(path_in)   
        # change from BGR to GRAYSCALE 
        img = compress(img,path_tmp_file)
        # save as png
        cv.imwrite(path_out, img, [cv.IMWRITE_PNG_COMPRESSION, PNG_SAVING_COMPRESSION]) 
        if debug: 
            # calc time
            compression_time = time.time()-time_start
            # calc  file sizes
            initial_size = os.stat(path_in).st_size
            compressed_size = os.stat(path_out).st_size
            # print file info
            print(
            f'''{i+1}/{len(os.listdir(PATH_INPUT))} 
            File {file} size: {initial_size/1e3:.2f} KB -> {compressed_size/1e3:.2f} KB [{compressed_size/initial_size*100:.2f}%]
            Compressed in {compression_time} s
            ''')

            size_all_initial += initial_size
            size_all_compressed += compressed_size
            list_compression_times.append(compression_time)

    if debug:
        print(f'''
        Processed {len(os.listdir(pth_input))} files. 
        Size: {size_all_initial/1e6:.2f} MB -> {size_all_compressed/1e6:.2f} MB [{size_all_compressed/size_all_initial*100:.2f}%]
        Mean time: {np.mean(np.array(list_compression_times))}
        ''')


compress_images_in_dir(PATH_INPUT,PATH_OUTPUT, debug=False)