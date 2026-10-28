"""Focused protocol checks for the independent hospital experiment."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("exp5_camelyon", ROOT/"exp5_camelyon.py")
exp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exp
spec.loader.exec_module(exp)


def fixture():
    rows = []
    for center, patient, partition in [(0,4,"source_train"),(3,64,"source_train"),(4,84,"source_train"),
                                        (1,24,"secondary_test"),(2,44,"primary_test")]:
        for i in range(6):
            rows.append(dict(center=center,patient=patient,node=0,slide=center*10,label=i%2,
                             sample_id=f"patient{patient}_x{i}",partition=partition,group_id=f"patient:{patient:03d}"))
    return pd.DataFrame(rows)


class HospitalDecoding(unittest.TestCase):
    def test_global_names(self):
        result, names = exp.decode_hospitals([0,3,4],{"names":list(exp.HOSPITAL_NAMES)})
        np.testing.assert_array_equal(result,[0,3,4])

    def test_reencoded_source_subset(self):
        result, names = exp.decode_hospitals([2,0,1],{"names":["train-center1","train-center2","train-center3"]})
        np.testing.assert_array_equal(result,[4,0,3])

    def test_dict_names(self):
        result, _ = exp.decode_hospitals([0],{"names":{"0":"test-center"}})
        np.testing.assert_array_equal(result,[2])

    def test_unknown_names_rejected(self):
        with self.assertRaises(ValueError):
            exp.decode_hospitals([0],{"names":["hospital-A"]})

    def test_missing_names_rejected(self):
        with self.assertRaises(ValueError):
            exp.decode_hospitals([0],{})

    def test_encoding_outside_support(self):
        with self.assertRaises(ValueError):
            exp.decode_hospitals([3],{"names":["test-center"]})


class ProspectiveSampling(unittest.TestCase):
    def test_label_blind(self):
        frame=fixture()
        for i in range(900):
            frame.loc[len(frame)]={**frame.iloc[0].to_dict(),"sample_id":f"extra{i}","label":i%2}
        first=exp.select_cohort(frame)
        frame["label"]=1-frame["label"]
        second=exp.select_cohort(frame.sample(frac=1,random_state=19))
        self.assertEqual(first["sample_id"].tolist(),second["sample_id"].tolist())
        self.assertEqual(len(first[first["patient"]==4]),400)

    def test_patient_cap_spans_multiple_nodes(self):
        frame=fixture()
        many=[]
        for i in range(1000):
            many.append({**frame.iloc[0].to_dict(),"node":i%2,"sample_id":f"two_nodes{i}"})
        selected=exp.select_cohort(pd.concat([frame,pd.DataFrame(many)]))
        self.assertEqual(len(selected[selected["patient"]==4]),400)

    def test_unused_id_validation_excluded(self):
        frame=fixture()
        frame.loc[len(frame)]={**frame.iloc[0].to_dict(),"sample_id":"unused","partition":"unused_source_id_validation"}
        self.assertNotIn("unused",exp.select_cohort(frame)["sample_id"].tolist())

    def test_exact_identity_format(self):
        self.assertEqual(exp.patch_identity(4,2,123,456),"patch_patient_004_node_2_x_123_y_456.png")

    def test_rank_prefix_exact(self):
        import hashlib
        expected=hashlib.sha256(b"camelyon17-patient-cap400-seed2027|abc").hexdigest()
        self.assertEqual(exp.sampling_rank("abc"),expected)


class PatientPartitionGuards(unittest.TestCase):
    def test_valid_cohort(self):
        exp.validate_cohort(fixture())

    def test_missing_target_class_stops(self):
        frame=fixture()
        frame.loc[frame["center"]==2,"label"]=0
        with self.assertRaisesRegex(ValueError,"lacks a class"):
            exp.validate_cohort(frame)

    def test_wrong_patient_center_stops(self):
        frame=fixture()
        frame.loc[0,"patient"]=44
        with self.assertRaisesRegex(ValueError,"Hospital/patient"):
            exp.validate_cohort(frame)

    def test_wrong_slide_center_stops(self):
        frame=fixture()
        frame.loc[0,"slide"]=21
        with self.assertRaisesRegex(ValueError,"Hospital/slide"):
            exp.validate_cohort(frame)

    def test_duplicate_patch_identity_stops(self):
        frame=fixture()
        frame.loc[1,"sample_id"]=frame.loc[0,"sample_id"]
        with self.assertRaisesRegex(ValueError,"duplicate patch"):
            exp.validate_cohort(frame)

    def test_source_folds_target_free(self):
        frame=fixture()
        train, folds=exp.source_folds(frame)
        self.assertEqual(set(frame.iloc[train]["center"]),{0,3,4})
        for tr,va in folds:
            self.assertFalse(set(tr)&set(va))
            self.assertFalse(set(frame.iloc[tr]["patient"])&set(frame.iloc[va]["patient"]))
            self.assertEqual(len(set(frame.iloc[va]["center"])),1)

    def test_feature_zero_rejected(self):
        with self.assertRaises(ValueError):
            exp.validate_features(np.zeros((3,512),dtype=np.float32),3)

    def test_feature_valid(self):
        f=np.ones((3,512),dtype=np.float32)/np.sqrt(512)
        exp.validate_features(f,3)

    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            exp.guarded(ROOT/"not_exp5.csv")

    def test_target_scoring_missing_freeze_rejected(self):
        with patch.object(exp,"OUT",ROOT/"exp5_camelyon/nonexistent_test_gate"):
            with self.assertRaisesRegex(ValueError,"forbidden"):
                exp.evaluate_frozen_bank(None,None)


if __name__=="__main__":
    unittest.main(verbosity=2)
