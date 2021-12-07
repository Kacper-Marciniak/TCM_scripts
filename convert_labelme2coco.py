# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = r"D:\Konrad\TCM_scan\training_extraction\data\train"

# set path for coco json to be saved
save_json_path = r"D:\Konrad\TCM_scan\training_extraction\data\annotations_train.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)
print("Finished")
