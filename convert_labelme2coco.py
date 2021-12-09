# search-ms:displayname=Search%20Results%20in%20Local%20Disk%20(C%3A)&crumb=location:C%3A%5C\labelme2coco
# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = r"D:\Konrad\TCM_scan\traning_segmentation\data\train"

# set path for coco json to be saved
#save_json_path = r"D:\Konrad\TCM_scan\training_extraction\data\annotations_train.json"
save_json_path = r"D:\Konrad\TCM_scan\traning_segmentation\data\annotations\teeth_coco2.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)
print("Finished")
