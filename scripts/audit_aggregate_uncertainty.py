"""Paired group uncertainty for sequence and patient-macro patch readouts."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from report_review_revision import confusion_ba, group_confusions, pair_counts, ranking_samples
from review_protocol import grids, read_csv, sha256, write_csv, write_json


def audit(report, output, attempts=2000, seed=2027):
    groups = read_csv(report / "group_metrics_by_seed.csv")
    sequence_path = report / "sequence_predictions.csv"
    sequence_means_path = report / "sequence_metrics_seed_means.csv"
    patient_means_path = report / "patient_macro_seed_means.csv"
    has_sequences = any(row["task"].startswith("polypgen") for row in groups)
    has_patients = any(row["task"].startswith("camelyon") for row in groups)
    if has_sequences and (not sequence_path.is_file() or not sequence_means_path.is_file()):
        raise ValueError("A PolypGen report requires its sequence predictions and sequence summaries")
    if has_patients and not patient_means_path.is_file():
        raise ValueError("A CAMELYON report requires its patient-macro patch summary")
    entities = read_csv(sequence_path) if sequence_path.is_file() else []
    patient_means = {(r["task"],r["protocol"],r["center"],r["method"]):r
                     for r in (read_csv(patient_means_path) if patient_means_path.is_file() else [])}
    sequence_means = {(r["task"],r["protocol"],r["center"],r["method"]):r
                      for r in (read_csv(sequence_means_path) if sequence_means_path.is_file() else [])}
    if not has_sequences and not has_patients:
        raise ValueError("No supported sequence or patient-macro endpoints are present")
    support = {}
    for row in groups:
        support.setdefault((row["center"],row["task"]),set()).add(row["group_id"])
    generator = np.random.default_rng(seed)
    draws = {}
    for center in ("C2","C4","C5","C6","1","2"):
        for (site,task),members in sorted(support.items()):
            if site != center:
                continue
            members = tuple(sorted(members))
            if (center,members) not in draws:
                draws[center,members] = generator.multinomial(len(members),np.full(len(members),1/len(members)),size=attempts)
    cells, point, samples = defaultdict(list), {}, {}
    for row in groups:
        if row["task"].startswith("camelyon"):
            cells[("patient_macro_patch",row["task"],row["protocol"],row["center"],row["method"])].append(row)
    for row in entities:
        cells[("sequence_presence",row["task"],row["protocol"],row["center"],row["method"])].append(row)
    for key,rows in cells.items():
        endpoint,task,protocol,center,method = key
        members = sorted({r["group_id"] for r in rows})
        multiplicity = draws[center,tuple(members)]
        seeds = sorted({r["seed"] for r in rows})
        if len(seeds) != 10:
            raise ValueError("Aggregate audit requires ten fitted seeds")
        if endpoint == "patient_macro_patch":
            values=[]
            for member in members:
                local=[r for r in rows if r["group_id"]==member]
                if len(local)!=10 or any(not r[m] for r in local for m in ("balanced_accuracy","auroc")):
                    raise ValueError("This patient-macro audit requires both patch classes per patient")
                values.append([np.mean([float(r[m]) for r in local]) for m in ("balanced_accuracy","auroc")])
            values=np.asarray(values)
            estimate,bootstrap=np.mean(values,axis=0),multiplicity@values/len(members)
            reference=patient_means[task,protocol,center,method]
        else:
            confusions,wins,pairs=[],[],None
            for fitting_seed in seeds:
                local=[r for r in rows if r["seed"]==fitting_seed]
                if len(local)!=len(members) or any(r["prediction"]=="" for r in local):
                    raise ValueError("Missing source-thresholded sequence decisions")
                order,confusion=group_confusions(local)
                order2,counts,concordance=pair_counts([{**r,"probability_class_1":r["score"]} for r in local])
                if order!=members or order2!=members or pairs is not None and not np.array_equal(pairs,counts):
                    raise ValueError("Unpaired sequence support")
                pairs=counts
                confusions.append(confusion)
                wins.append(concordance)
            confusion=np.mean(confusions,axis=0)
            concordance=np.mean(wins,axis=0)[None]
            estimate=np.array([confusion_ba(confusion,np.ones((1,len(members))))[0],
                               ranking_samples(pairs,concordance,np.ones((1,len(members))))[0,0,0]])
            bootstrap=np.column_stack((confusion_ba(confusion,multiplicity),ranking_samples(pairs,concordance,multiplicity)[:,0,0]))
            reference=sequence_means[task,protocol,center,method]
        for i,metric in enumerate(("balanced_accuracy","auroc")):
            if reference[metric]:
                np.testing.assert_allclose(estimate[i],float(reference[metric]),atol=1e-12,rtol=0)
            elif np.isfinite(estimate[i]):
                raise ValueError("Unsupported aggregate unexpectedly has a defined point metric")
            point[key+(metric,)]=estimate[i]
            samples[key+(metric,)]=bootstrap[:,i]
    records=[]
    for endpoint,task,protocol in sorted({key[:3] for key in cells}):
        if protocol=="factorial":
            continue
        methods=list(grids(protocol))
        for metric in ("balanced_accuracy","auroc"):
            centers=sorted({key[3] for key in cells if key[:3]==(endpoint,task,protocol)})
            eligible=[c for c in centers if np.isfinite(point[endpoint,task,protocol,c,"cbmp",metric])]
            if eligible:
                for method in methods:
                    key=(endpoint,task,protocol,"equal_center_mean",method,metric)
                    point[key]=np.mean([point[endpoint,task,protocol,c,method,metric] for c in eligible])
                    samples[key]=np.mean([samples[endpoint,task,protocol,c,method,metric] for c in eligible],axis=0)
                centers.append("equal_center_mean")
            for center in centers:
                for method in methods:
                    if method=="cbmp":continue
                    a=(endpoint,task,protocol,center,"cbmp",metric)
                    b=(endpoint,task,protocol,center,method,metric)
                    delta=samples[a]-samples[b]
                    valid=delta[np.isfinite(delta)]
                    ci=np.quantile(valid,[.025,.975]) if len(valid) else (None,None)
                    difference=point[a]-point[b]
                    records.append({"endpoint":endpoint,"task":task,"protocol":protocol,"center":center,
                                    "comparison":"cbmp_minus_"+method,"metric":metric,
                                    "difference":float(difference) if np.isfinite(difference) else None,
                                    "ci_low":ci[0],"ci_high":ci[1],"valid_draws":len(valid),"attempts":attempts,
                                    "eligible_centers_for_mean":"|".join(eligible),"fitting_seeds":10})
    if output.exists():raise FileExistsError("Choose a new aggregate-audit directory")
    output.mkdir(parents=True)
    write_csv(output/"paired_aggregate_intervals.csv",records)
    write_json(output/"provenance.json",{"report_provenance_sha256":sha256(report/"provenance.json"),
               "input_hashes":{path.name:sha256(path) for path in
                    (report/"group_metrics_by_seed.csv", sequence_path, sequence_means_path, patient_means_path)
                    if path.is_file()},
               "attempts":attempts,"seed":seed,"auditor_sha256":sha256(Path(__file__)),
               "scope":"observed recorded groups, fixed fitted heads; no clinical patient labels inferred",
               "point_estimates_reconstruct_report":True})
    print(f"Reconstructed aggregate points and reported {len(records)} paired intervals")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    audit(args.report_dir.resolve(),args.output_dir.resolve())
