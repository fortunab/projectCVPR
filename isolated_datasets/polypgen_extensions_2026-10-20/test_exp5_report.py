"""Meaningful bootstrap and reconstruction numerical checks."""
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import exp5_report as rep
import exp5_camelyon as exp


class PatientBootstrapNumerics(unittest.TestCase):
    def test_exact_weighted_tie_auc(self):
        y=np.array([0,1,1,0,0,1])
        p=np.array([.1,.5,.5,.9,.2,.9])
        gi=np.array([0,0,1,1,2,2])
        counts=np.array([[1,1,1],[0,3,0],[3,0,2],[1,2,3]])
        result=rep.weighted_auc_sorted(y,p,gi,counts)
        for values,auc in zip(counts,result):
            expected=roc_auc_score(y,p,sample_weight=values[gi])
            self.assertAlmostEqual(auc,expected,places=13)

    def test_class_incomplete_auc_undefined(self):
        result=rep.weighted_auc_sorted(np.array([0,1]),np.array([.2,.8]),np.array([0,1]),np.array([[3,0],[0,3]]))
        self.assertTrue(np.isnan(result).all())

    def test_patient_bootstrap_all_ones_reconstructs_mean_seed_metrics(self):
        records=[]
        seedmetrics=[]
        for seed in exp.SEEDS:
            labels=np.array([0,1,0,1,0,1])
            probs=np.array([.1,.8,.7,.6,.2,.4])+(seed%5)*.01
            prob=np.column_stack([1-probs,probs])
            seedmetrics.append(exp.all_metrics(labels,prob))
            for i,(y,p) in enumerate(zip(labels,probs)):
                records.append({"seed":seed,"sample_id":f"patch{i}","patient":i//2,"label":int(y),"probability_class_1":p,"prediction":int(p>.5)})
        result=rep.bootstrap_method(pd.DataFrame(records),np.ones((1,3),dtype=int),[0,1,2])
        for metric in rep.METRICS:
            self.assertAlmostEqual(result[metric][0],np.mean([v[metric] for v in seedmetrics]),places=12)

    def test_patient_draw_duplicates_whole_patient(self):
        labels=np.array([0,1,0,1])
        probs=np.array([.1,.6,.9,.8])
        rows=[]
        for seed in exp.SEEDS:
            for i,(y,p) in enumerate(zip(labels,probs)):
                rows.append({"seed":seed,"sample_id":f"p{i}","patient":i//2,"label":int(y),"probability_class_1":p,"prediction":int(p>.5)})
        result=rep.bootstrap_method(pd.DataFrame(rows),np.array([[2,0]]),[0,1])
        self.assertEqual(result["balanced_accuracy"][0],1)
        self.assertEqual(result["auroc"][0],1)


if __name__=="__main__":
    unittest.main(verbosity=2)
