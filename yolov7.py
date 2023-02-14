from mymodel.detect_new import detect
from convertYoloToFormula import convertYoloToSingleFormula


def run_yolo(image_name):
    detect('best.pt', image_name, 640, conf_thres=0.25, iou_thres=0.45)
    convertYoloToSingleFormula(image_name)
    return image_name


if __name__ == '__main__':
    run_yolo('test.jpg')
