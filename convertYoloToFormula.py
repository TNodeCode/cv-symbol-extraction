import yaml
import torch
from pathlib import Path


def overlap(boxes1, boxes2):
    # top left corners of all combinations
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

    # bottom right corners of all combinations
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # width and hight of overlap area of boxes1 and boxes2.
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]#

    boxes1Area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2Area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    iou = inter / (boxes1Area + boxes2Area - inter)

    return iou


def coordinatesConverter(bboxes, img_height, img_width):
    """
    Converts BBoxes from xywh (standardized) in xyxy
    """
    bboxes[:, 0] *= 2 * img_width
    bboxes[:, 1] *= 2 * img_height

    bboxes[:, 2] *= img_width
    bboxes[:, 3] *= img_height

    bboxes[:, 0] = (bboxes[:, 0] - bboxes[:, 2]) / 2
    bboxes[:, 1] = (bboxes[:, 1] - bboxes[:, 3]) / 2
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    return bboxes


def convertYoloToSingleFormula():
    # replace with Path to custom_data.yaml for encoding
    #with open("C:/Users/benwe/Downloads/custom_data.yaml", "r") as file:
    #    data = yaml.safe_load(file)["names"]
    #    labels_encode = {}
    #    for i in range(len(data)):
    #        labels_encode[data[i]] = i

    fomrula_code = 0

    yolo_output = Path("detections")
    if not yolo_output.exists():
        yolo_output.mkdir()
    yolo_labels_path = yolo_output / "labels"

    formulaLabels_path = yolo_output / "formulaLabels"
    if not formulaLabels_path.exists():
        formulaLabels_path.mkdir()

    yolo_labels = [file for file in yolo_labels_path.glob("*.txt")]

    output_counter = 0
    for file in yolo_labels:
        labels = open(file, "r").readlines()
        labels = list(map(lambda x: torch.Tensor(
            list(map(float, x.strip('\n').split(" ")))), labels))
        formulas_in_image = [box for box in labels if box[0] == fomrula_code]

        boxes2 = torch.stack(labels)[:, 1:]
        boxes2 = coordinatesConverter(boxes2, 640, 640)

        for formula in formulas_in_image:
            boxes1 = formula[1:].clone().reshape((1, 4))
            boxes1 = coordinatesConverter(boxes1, 640, 640)

            iou = overlap(boxes1, boxes2)
            _, boxes_in_formula_idx = torch.where(iou > 0)

            boxes_in_formula = torch.stack(
                labels)[boxes_in_formula_idx].tolist()
            boxes_in_formula = [[int(box[0]), *box[1:]]
                                for box in boxes_in_formula if box[0] != fomrula_code]

            label_txt = open(formulaLabels_path / f"{output_counter}.txt", 'a')
            label_txt.write(f"{formula[0]} {formula[1]} {formula[2]} {formula[3]} {formula[4]}\n")

            for line_ in boxes_in_formula:
                label_txt.write(
                    f"{line_[0]} {line_[1]} {line_[2]} {line_[3]} {line_[4]}")
                label_txt.write("\n")
            label_txt.close()
            output_counter += 1

            # # For visualization
            # print(boxes_in_formula)
            # for i in boxes_in_formula_idx:
            #     box = boxes2[i]
            #     plt.gca().add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='g', facecolor='none'))

            # plt.gca().add_patch(Rectangle((boxes1[0,0], boxes1[0,1]), boxes1[0,2]-boxes1[0,0], boxes1[0,3]-boxes1[0,1], linewidth=1, edgecolor='b', facecolor='none'))
            # img = torch.zeros((640, 640))
            # plt.imshow(img)
            # plt.show()