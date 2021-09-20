import glob
import fiftyone as fo

images_patt = r'C:\Users\Konrad\tcm_scan\20210621_092043_data\images'

# Ex: your custom label format
annotations = {"images": [{"file_name": "tooth_0.png", "height": 3840, "width": 2748, "id": 0}, {"file_name": "tooth_1.png", "height": 3840, "width": 2748, "id": 1}, {"file_name": "tooth_11.png", "height": 3840, "width": 2748, "id": 11}, {"file_name": "tooth_12.png", "height": 3840, "width": 2748, "id": 12}, {"file_name": "tooth_13.png", "height": 3840, "width": 2748, "id": 13}, {"file_name": "tooth_14.png", "height": 3840, "width": 2748, "id": 14}, {"file_name": "tooth_15.png", "height": 3840, "width": 2748, "id": 15}, {"file_name": "tooth_16.png", "height": 3840, "width": 2748, "id": 16}, {"file_name": "tooth_17.png", "height": 3840, "width": 2748, "id": 17}, {"file_name": "tooth_18.png", "height": 3840, "width": 2748, "id": 18}, {"file_name": "tooth_19.png", "height": 3840, "width": 2748, "id": 19}, {"file_name": "tooth_2.png", "height": 3840, "width": 2748, "id": 2}, {"file_name": "tooth_21.png", "height": 3840, "width": 2748, "id": 21}, {"file_name": "tooth_22.png", "height": 3840, "width": 2748, "id": 22}, {"file_name": "tooth_23.png", "height": 3840, "width": 2748, "id": 23}, {"file_name": "tooth_3.png", "height": 3840, "width": 2748, "id": 3}, {"file_name": "tooth_4.png", "height": 3840, "width": 2748, "id": 4}, {"file_name": "tooth_5.png", "height": 3840, "width": 2748, "id": 5}, {"file_name": "tooth_6.png", "height": 3840, "width": 2748, "id": 6}, {"file_name": "tooth_7.png", "height": 3840, "width": 2748, "id": 7}, {"file_name": "tooth_8.png", "height": 3840, "width": 2748, "id": 8}, {"file_name": "tooth_9.png", "height": 3840, "width": 2748, "id": 9}], "type": "instances", "annotations": [{"area": 398944, "iscrowd": 0, "image_id": 0, "bbox": [1276, 1545, 728, 548], "category_id": 0, "id": 1, "ignore": 0, "segmentation": []}, {"area": 361696, "iscrowd": 0, "image_id": 1, "bbox": [1287, 1689, 712, 508], "category_id": 0, "id": 2, "ignore": 0, "segmentation": []}, {"area": 282396, "iscrowd": 0, "image_id": 11, "bbox": [1386, 1873, 699, 404], "category_id": 0, "id": 3, "ignore": 0, "segmentation": []}, {"area": 305300, "iscrowd": 0, "image_id": 12, "bbox": [1391, 1875, 710, 430], "category_id": 0, "id": 4, "ignore": 0, "segmentation": []}, {"area": 320052, "iscrowd": 0, "image_id": 13, "bbox": [1397, 1893, 716, 447], "category_id": 0, "id": 5, "ignore": 0, "segmentation": []}, {"area": 299600, "iscrowd": 0, "image_id": 14, "bbox": [1405, 1901, 700, 428], "category_id": 0, "id": 6, "ignore": 0, "segmentation": []}, {"area": 308136, "iscrowd": 0, "image_id": 15, "bbox": [1415, 1908, 694, 444], "category_id": 0, "id": 7, "ignore": 0, "segmentation": []}, {"area": 318208, "iscrowd": 0, "image_id": 16, "bbox": [1421, 1913, 704, 452], "category_id": 0, "id": 8, "ignore": 0, "segmentation": []}, {"area": 302996, "iscrowd": 0, "image_id": 17, "bbox": [1421, 1955, 718, 422], "category_id": 0, "id": 9, "ignore": 0, "segmentation": []}, {"area": 311750, "iscrowd": 0, "image_id": 18, "bbox": [1427, 1953, 725, 430], "category_id": 0, "id": 10, "ignore": 0, "segmentation": []}, {"area": 304432, "iscrowd": 0, "image_id": 19, "bbox": [1430, 1955, 718, 424], "category_id": 0, "id": 11, "ignore": 0, "segmentation": []}, {"area": 349896, "iscrowd": 0, "image_id": 2, "bbox": [1297, 1704, 717, 488], "category_id": 0, "id": 12, "ignore": 0, "segmentation": []}, {"area": 319846, "iscrowd": 0, "image_id": 21, "bbox": [1445, 1979, 722, 443], "category_id": 0, "id": 13, "ignore": 0, "segmentation": []}, {"area": 308066, "iscrowd": 0, "image_id": 22, "bbox": [1451, 2005, 737, 418], "category_id": 0, "id": 14, "ignore": 0, "segmentation": []}, {"area": 313900, "iscrowd": 0, "image_id": 23, "bbox": [1458, 2013, 730, 430], "category_id": 0, "id": 15, "ignore": 0, "segmentation": []}, {"area": 343650, "iscrowd": 0, "image_id": 3, "bbox": [1304, 1728, 725, 474], "category_id": 0, "id": 16, "ignore": 0, "segmentation": []}, {"area": 356400, "iscrowd": 0, "image_id": 4, "bbox": [1310, 1740, 720, 495], "category_id": 0, "id": 17, "ignore": 0, "segmentation": []}, {"area": 365568, "iscrowd": 0, "image_id": 5, "bbox": [1319, 1729, 714, 512], "category_id": 0, "id": 18, "ignore": 0, "segmentation": []}, {"area": 315530, "iscrowd": 0, "image_id": 6, "bbox": [1344, 1756, 695, 454], "category_id": 0, "id": 19, "ignore": 0, "segmentation": []}, {"area": 308136, "iscrowd": 0, "image_id": 7, "bbox": [1350, 1783, 694, 444], "category_id": 0, "id": 20, "ignore": 0, "segmentation": []}, {"area": 339450, "iscrowd": 0, "image_id": 8, "bbox": [1342, 1800, 730, 465], "category_id": 0, "id": 21, "ignore": 0, "segmentation": []}, {"area": 326755, "iscrowd": 0, "image_id": 9, "bbox": [1355, 1818, 715, 457], "category_id": 0, "id": 22, "ignore": 0, "segmentation": []}], "categories": [{"supercategory": "none", "id": 0, "name": "tooth"}]}

# Create samples for your data
samples = []
for filepath in glob.glob(images_patt):
    sample = fo.Sample(filepath=filepath)

    # Convert detections to FiftyOne format
    detections = []
    for obj in annotations[filepath]:
        label = obj["label"]

        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        bounding_box = obj["bbox"]

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )

    # Store detections in a field name of your choice
    sample["ground_truth"] = fo.Detections(detections=detections)

    samples.append(sample)

# Create dataset
dataset = fo.Dataset("my-detection-dataset")
dataset.add_samples(samples)