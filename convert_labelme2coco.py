# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = r"D:\Konrad\TCM_scan\traning_segmentation\data\train"

# set path for coco json to be saved
save_json_path = r"D:\Konrad\TCM_scan\traning_segmentation\data\annotations\teeth_coco2.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)
