
# import functions
from labelme2coco import get_coco_from_labelme_folder, save_json

base_folder = r"D:\Konrad\TCM_scan\traning_segmentation/"


# set labelme training data directory
labelme_train_folder = base_folder + "data/train"

# set labelme validation data directory
labelme_val_folder = base_folder + "data/val"

# set labelme validation data directory
labelme_test_folder = base_folder + "data/test"

# set path for coco json to be saved
export_dir = base_folder + "data/annotations/data_"

# create train coco object
train_coco = get_coco_from_labelme_folder(labelme_train_folder)
# export train coco json 
save_json(train_coco.json, export_dir+"train.json")

# create val coco object
val_coco = get_coco_from_labelme_folder(labelme_val_folder, coco_category_list = train_coco.json_categories)
# export val coco json
save_json(val_coco.json, export_dir+"val.json")

# create val coco object
test_coco = get_coco_from_labelme_folder(labelme_test_folder, coco_category_list = train_coco.json_categories)
# export val coco json
save_json(test_coco.json, export_dir+"test.json")