# import package
import labelme2coco



# set directory that contains labelme annotations and image files
labelme_folder = r"H:\Konrad\tcm_scan\20210621_092043\otsu_tooth"

# set path for coco json to be saved
save_json_path = r"H:\Konrad\tcm_scan\20210621_092043\annotations\teeth_coco.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)