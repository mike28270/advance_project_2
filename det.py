from detection.yolov5.detect import run as det
from ontology.utils.ontology import Ontology, Labels
import json


def process(config_file: str, source="dataset/machine_detection-4/test/images", name="detection"):
    model = det(
        weights="detection/models/aug2_yolov5l/weights/best.pt",
        data="detection/dataset/data.yaml",
        source=source,
        project="results",
        name=name,
        conf_thres=0.25,
        iou_thres=0.6,
        line_thickness=1,
        nosave=False,
        save_json=True,
        exist_ok=True
    )

    labels_list = Labels().load_label_from_model(model)

    mason = Ontology("morenap/onto_repo", "mason.owl")
    mason.get_onto
    mason.semetic_search(labels_list, save_path="results")


if __name__ == "__main__":
    process(
        config_file="detection/config.yaml",
        source="morenap/media/images", 
        name="detection")
    