from torchmetrics.detection.mean_ap import MeanAveragePrecision
import json
from torch import Tensor
import pandas as pd
from evalutils.scorers import score_detection
import os
import yaml
import numpy as np

from typing import Dict

class MIDOG2021Evaluation():
    
    def __init__(self,path):
        self._predictions_file = os.path.join(path, 'mitotic-figures.json')
        self._gt_file = os.path.join(path, 'ground-truth.json')
        self._output_file = os.path.join(path, 'metrics.json')
        with open(os.path.join(path, "config.yaml"), 'r') as stream:
            self.config = yaml.safe_load(stream)
        self.val_gt, self.test_gt = {}, {}
        self.load_gt()

        self.cases = pd.read_csv('datasets_xvalidation.csv', delimiter=";")
        self.case_to_tumor = {'%03d.tiff' % d.loc['Slide'] : d.loc['Tumor'] for _, d in self.cases.iterrows()}

    def load_gt(self):
        self.gt = json.load(open(self._gt_file,'r'))
        val_files = json.loads(self.config['x-validation']['valid'])
        for key, value in self.gt.items():
            if key in val_files:
                self.val_gt.update({key:value})
            else:
                self.test_gt.update({key:value})
        
    def load_predictions(self):
        predictions_json = json.load(open(self._predictions_file,'r'))
        predictions={}
        for fname, pred in predictions_json.items():
            if (fname not in self.gt):
                print('Warning: Found predictions for image ',fname,'which is not part of the ground truth.')
                continue

            if 'points' not in pred:
                    print('Warning: Wrong format. Field points is not part of detections.')
                    continue
            points=[]

            for point in pred['points']:
                    detected_class = 1 if 'name' not in point or point['name']=='mitotic figure' else 0
                    detected_thr   = 0.5 if 'probability' not in point else point['probability']

                    if 'name' not in point:
                        print('Warning: Old format. Field name is not part of detections.')

                    if 'probability' not in point:
                        print('Warning: Old format. Field probability is not part of detections.')
                    
                    if 'point' not in point:
                        print('Warning: Point is not part of points structure.')
                        continue

                    points.append([*point['point'][0:3], detected_class, detected_thr])

            predictions[fname]=points
        self.predictions=predictions

    @property
    def _metrics(self) -> Dict:
        """ Returns the calculated case and aggregate results """
        return {
            "case": self._case_results,
            "aggregates": self._aggregate_results,
        }        
    def score(self, val=True, det=0.5):
        cases = list(self.val_gt.keys()) if val else list(self.test_gt.keys())
        self._case_results={}
        self.map_metric = MeanAveragePrecision()
        tumor_types =  list(self.cases[self.cases['Slide'].isin([int(c[:3]) for c in cases])]['Tumor'].unique())
        self.per_tumor_map_metric = {d: MeanAveragePrecision() for d in tumor_types}
        for idx, case in enumerate(cases):
            if case not in self.predictions:
                print('Warning: No prediction for file: ',case)
                continue

            # Filter out all predictions with class==0, retain predictions with class==1
            filtered_predictions = [(x,y,0) for x,y,z,cls,sc in self.predictions[case] if cls==1 and sc > det]

            bbox_size = 0.01125 # equals to 7.5mm distance for horizontal distance at 0.5 IOU

            pred_dict = [{'boxes': Tensor([[x-bbox_size,y-bbox_size, x+bbox_size, y+bbox_size] for (x,y,z,_,_) in self.predictions[case]]), 
                         'labels': Tensor([1,]*len(self.predictions[case])),
                         'scores': Tensor([sc for (x,y,z,sc,_) in self.predictions[case]])}]
            target_dict = [{'boxes': Tensor([[x-bbox_size,y-bbox_size, x+bbox_size, y+bbox_size] for (x,y,z) in self.gt[case]]),
                           'labels' : Tensor([1,]*len(self.gt[case]))}]

            self.map_metric.update(pred_dict,target_dict)
            self.per_tumor_map_metric[self.case_to_tumor[case]].update(pred_dict,target_dict)

            sc = score_detection(ground_truth=self.gt[case],predictions=filtered_predictions,radius=7.5E-3)._asdict()
            self._case_results[case] = sc

        self._aggregate_results = self.score_aggregates()

    def save(self):
        with open(self._output_file, "w") as f:
                    f.write(json.dumps(self._metrics))        
    def evaluate(self):
        self.load_predictions()
        thresh = self.find_threshold()
        self.score(val=False, det=thresh)
        self._aggregate_results['det_threshold'] = thresh
        self.save()

    def find_threshold(self):
        f1_scores = {}
        for thresh in np.arange(0.5, 1, 0.01):
            self.score(det=thresh)
            f1_scores.update({thresh: self._aggregate_results['f1_score']})
        return max(f1_scores, key=f1_scores.get)

    def score_aggregates(self):
        # per tumor stats
        per_tumor = {d : {'tp': 0, 'fp':0, 'fn':0} for d in self.per_tumor_map_metric}

        tp,fp,fn = 0,0,0
        for s in self._case_results:
            tp += self._case_results[s]["true_positives"]            
            fp += self._case_results[s]["false_positives"]            
            fn += self._case_results[s]["false_negatives"]            

            per_tumor[self.case_to_tumor[s]]['tp'] += self._case_results[s]["true_positives"] 
            per_tumor[self.case_to_tumor[s]]['fp'] += self._case_results[s]["false_positives"] 
            per_tumor[self.case_to_tumor[s]]['fn'] += self._case_results[s]["false_negatives"] 

        aggregate_results=dict()

        eps = 1E-6

        aggregate_results["precision"] = tp / (tp + fp + eps)
        aggregate_results["recall"] = tp / (tp + fn + eps)
        aggregate_results["f1_score"] = (2 * tp + eps) / ((2 * tp) + fp + fn + eps)

        metrics_values = self.map_metric.compute()
        aggregate_results["mAP"] = metrics_values['map_50'].tolist()

        for tumor in per_tumor:
            aggregate_results[f'{tumor}_precision'] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fp'] + eps)
            aggregate_results[f'{tumor}_recall'] = per_tumor[tumor]['tp'] / (per_tumor[tumor]['tp'] + per_tumor[tumor]['fn'] + eps)
            aggregate_results[f'{tumor}_f1'] = (2 * per_tumor[tumor]['tp'] + eps) / ((2 * per_tumor[tumor]['tp']) + per_tumor[tumor]['fp'] + per_tumor[tumor]['fn'] + eps)

            pt_metrics_values = self.per_tumor_map_metric[tumor].compute()
            aggregate_results[f"{tumor}_mAP"] = pt_metrics_values['map_50'].tolist()
        return aggregate_results

def evaluate(directory):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            with open(os.path.join(directory, dir, "files", "wandb-summary.json"), 'r') as f:
                data = json.load(f)
            # Training is done but model has not yet been evaluated
            if os.path.exists(os.path.join(directory, dir, "files", "mitotic-figures.json")):
                evaluation = MIDOG2021Evaluation(os.path.join(directory, dir, "files"))
                print("Evaluating", dir)
                evaluation.evaluate()
        break


