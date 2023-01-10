from typing import List, Set

import yaml
from yaml.loader import SafeLoader

import torch
from sentence_transformers import SentenceTransformer, util

from owlready2 import get_ontology, onto_path

import json

class Labels:
    def __init__(self) -> None:
        pass

    def _extract_label(self, objects: List) -> List[str]:
        return list(set([object.split('_')[0] for object in objects]))

    def load_label_from_yaml(self, path: str) -> List[str]:
        assert path.endswith(".yaml"), 'The file must be .yaml'
        data = yaml.load(open(path), Loader=SafeLoader)
        self.labels = self._extract_label(data.get('names'))
        return self.labels

    def load_label_from_model(self, model) -> List[str]:
        data = model.names
        self.labels = self._extract_label(data.values())
        return self.labels


class Ontology:
    def __init__(self, path: str, onto_name: str):
        onto_path.append(path)
        assert onto_name.endswith(".owl"), 'The file must be .owl'
        self.onto = get_ontology(onto_name)
        self.onto.load()

    @property
    def get_onto(self):
        return self.onto

    def get_classnames(self):
        return [str(a_class) for a_class in self.onto.classes()]

    def semetic_search(
            self, 
            labels: List[str],
            transformer_model: str = "all-MiniLM-L6-v2",
            closest: int = 3,
            save_path: str = None,
        ):
        embedder = SentenceTransformer(transformer_model)
        classnames = self.get_classnames()
        classnames_en = embedder.encode(classnames, convert_to_tensor=True)

        # Query sentences
        top_k = min(closest, len(classnames_en))
        search_result = {}
        for label in labels:
            label_en = embedder.encode(label, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the top scores
            cos_scores = util.cos_sim(label_en, classnames_en)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            top_dict = {}
            for score, idx in zip(top_results[0].tolist(), top_results[1].tolist()):
                top_dict[classnames[idx]] = score
            search_result[label] = top_dict
        if save_path is not None:
            with open(f"{save_path}/search.json", "w") as json_file:
                json.dump(search_result, json_file, indent=4)
        return search_result


if __name__ == "__main__":
    # print(ontology("morenap/onto_repo", "mason.owl").getclasses())
    pass