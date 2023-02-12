from detection.utils.inference import run as inference
from ontology.utils.ontology import Ontology, Labels
import json


def process(
    config_file: str, 
    weights="detection/models/aug2_yolov5l/weights/best.pt",
    data="detection/dataset/data.yaml",
    source="morenap/media/images",
    project="results",
    name="detection",
    conf_thres=0.25,
    iou_thres=0.6,
    line_thickness=1,
    nosave=False,
    save_json=True,
    exist_ok=True):

    model = inference(
        weights=weights,
        data=data,
        source=source,
        project=project,
        name=name,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        line_thickness=line_thickness,
        nosave=nosave,
        save_json=save_json,
        exist_ok=exist_ok
    )

    labels_list = Labels().load_label_from_model(model)

    mason = Ontology()
    mason.set_onto("morenap/onto_repo", "mason.owl")
    mason.get_onto
    mason.semetic_search(labels_list, save_path=f"{project}/{name}")


if __name__ == "__main__":
    process(
        weights="detection/models/whole_aug1_yolov5l/best.pt",
        config_file="detection/config.yaml",
        data="detection/dataset/data_whole.yaml",
        source="/Users/kritkorns/Mike/Jacob/AdvancedProject2/rawdata/machine/images/", 
        name="detection_whole",
        project="results")
    