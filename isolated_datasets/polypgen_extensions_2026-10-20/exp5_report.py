"""Validate and report the prespecified independent hospital transfer subset."""
from __future__ import annotations
import json
import os
import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/"code"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"exp5_camelyon/matplotlib_config"))
import torch  # Initialize native libraries before pandas/Arrow on Windows.
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import exp5_camelyon as exp

OUT = exp.OUT
ATTEMPTS = 2000
METRICS = ("balanced_accuracy","auroc","macro_f1","sensitivity","specificity","negative_log_likelihood","brier_score")
LABELS = {"linear_logistic":"Linear logistic", "single_centroid_group_weighted":"Patient-weighted centroid",
          "trainable_cosine":"Trained cosine", "single_centroid_image_weighted":"Image-weighted centroid",
          "single_centroid_center_balanced":"Hospital-balanced centroid", "multi_prototype_unbalanced":"Unbalanced multi-prototype",
          "linear_group_center_weighted":"Weighted linear", "cosine_group_center_weighted":"Weighted cosine", "cbmp":"CBMP"}


def weighted_auc_sorted(labels, probabilities, patient_indices, counts, batch_size=100):
    """All patient-bootstrap AUROCs with exact weighted tie handling."""
    order=np.argsort(probabilities,kind="stable")
    labels=labels[order]
    probs=probabilities[order]
    group_indices=patient_indices[order]
    starts=np.r_[0,np.flatnonzero(probs[1:] != probs[:-1])+1]
    output=np.full(len(counts),np.nan)
    for start in range(0,len(counts),batch_size):
        weights=counts[start:start+batch_size,group_indices]
        positive=np.add.reduceat(weights*labels[None,:],starts,axis=1)
        negative=np.add.reduceat(weights*(1-labels[None,:]),starts,axis=1)
        totalp=positive.sum(axis=1)
        totaln=negative.sum(axis=1)
        denominator=totalp*totaln
        numerator=np.sum(positive*(np.cumsum(negative,axis=1)-.5*negative),axis=1)
        output[start:start+len(weights)]=np.divide(numerator,denominator,out=np.full(len(weights),np.nan),where=denominator>0)
    return output


def bootstrap_method(rows, counts, group_order):
    """Same patient draw applied to each fixed fitting seed; average seed metrics."""
    arrays={k:[] for k in METRICS}
    for seed in exp.SEEDS:
        sub=rows[rows["seed"]==seed].sort_values("sample_id")
        group_map={patient:i for i,patient in enumerate(group_order)}
        gi=np.array([group_map[v] for v in sub["patient"]])
        y=sub["label"].to_numpy(dtype=int)
        pred=sub["prediction"].to_numpy(dtype=int)
        probs=sub["probability_class_1"].to_numpy()
        n=len(group_order)
        positive=np.bincount(gi,weights=y,minlength=n)
        negative=np.bincount(gi,weights=1-y,minlength=n)
        tp=np.bincount(gi,weights=y*pred,minlength=n)
        tn=np.bincount(gi,weights=(1-y)*(1-pred),minlength=n)
        patch_count=np.bincount(gi,minlength=n)
        # Reconstruct sklearn probability clipping for the saved float64 softmax.
        clipped=np.clip(probs,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
        nll=np.bincount(gi,weights=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)),minlength=n)
        brier=np.bincount(gi,weights=(probs-y)**2,minlength=n)
        p,nneg=counts@positive,counts@negative
        tps,tns=counts@tp,counts@tn
        denom=counts@patch_count
        sens=np.divide(tps,p,out=np.full(len(p),np.nan),where=p>0)
        spec=np.divide(tns,nneg,out=np.full(len(p),np.nan),where=nneg>0)
        arrays["balanced_accuracy"].append(.5*(sens+spec))
        arrays["sensitivity"].append(sens)
        arrays["specificity"].append(spec)
        fps=nneg-tns
        fns=p-tps
        f1pos=np.divide(2*tps,2*tps+fps+fns,out=np.zeros(len(p)),where=(2*tps+fps+fns)>0)
        f1neg=np.divide(2*tns,2*tns+fps+fns,out=np.zeros(len(p)),where=(2*tns+fps+fns)>0)
        arrays["macro_f1"].append(.5*(f1pos+f1neg))
        arrays["negative_log_likelihood"].append((counts@nll)/denom)
        arrays["brier_score"].append((counts@brier)/denom)
        auc=weighted_auc_sorted(y,probs,gi,counts)
        # An all-one patient multiplicity exactly reconstructs the ordinary AUROC.
        original=weighted_auc_sorted(y,probs,gi,np.ones((1,n),dtype=int))[0]
        if not np.isclose(original,roc_auc_score(y,probs),atol=1e-12):
            raise ValueError("Weighted AUROC implementation fails ordinary reconstruction")
        arrays["auroc"].append(auc)
    return {metric:np.mean(values,axis=0) for metric,values in arrays.items()}


def validate_predictions():
    from cbmp_evaluation import head_scores,probabilities
    manifest=pd.read_csv(OUT/"audit/image_manifest.csv")
    metrics=pd.read_csv(OUT/"evaluation/metrics_by_seed_hospital.csv",float_precision="round_trip")
    predictions=pd.read_csv(OUT/"evaluation/predictions.csv",float_precision="round_trip")
    if len(metrics)!=54 or set(metrics["seed"])!=set(exp.SEEDS) or metrics["method"].nunique()!=9:
        raise ValueError("Incomplete expected target metrics")
    if predictions.duplicated(["sample_id","method","seed"]).any():
        raise ValueError("Duplicate saved predictions")
    frozen=json.loads((OUT/"fits/source_heads_frozen.json").read_text())
    if len(frozen["models"])!=27:
        raise ValueError("Incomplete source freeze bank")
    with np.load(OUT/"features/resnet18_imagenet1k_v1.npz",allow_pickle=False) as cache:
        features=cache["features"]
        if cache["sample_ids"].tolist()!=manifest["sample_id"].tolist():
            raise ValueError("Feature cache sample order mismatch")
    checks=[]
    for model_entry in frozen["models"]:
        model_path=OUT/model_entry["path"]
        if exp.sha256(model_path)!=model_entry["sha256"]:
            raise ValueError("Saved model hash changed")
        if model_path.stat().st_mtime>frozen["created_unix_seconds"]:
            raise ValueError("Source model file was written after full-bank freeze")
        saved=joblib.load(model_path)
        for hospital in exp.TARGETS:
            idx=np.flatnonzero(manifest["center"].to_numpy()==hospital)
            rows=manifest.iloc[idx]
            sub=predictions[(predictions["hospital"]==hospital)&(predictions["seed"]==model_entry["seed"])&
                            (predictions["method"]==model_entry["method"])].set_index("sample_id").loc[rows["sample_id"]]
            y=rows["label"].to_numpy(dtype=int)
            if not np.array_equal(sub["label"],y) or not np.array_equal(sub["patient"],rows["patient"]):
                raise ValueError("Saved prediction metadata changed")
            score=head_scores(saved["model"],features[idx])
            prob=probabilities(score,saved["temperature"])
            if (not np.array_equal(sub["prediction"],np.argmax(prob,axis=1)) or
                not np.allclose(sub["probability_class_1"],prob[:,1],atol=2e-15,rtol=2e-15)):
                raise ValueError("Model cannot reconstruct saved probabilities")
            reconstructed=exp.all_metrics(y,prob)
            measured=metrics[(metrics["hospital"]==hospital)&(metrics["seed"]==model_entry["seed"])&
                             (metrics["method"]==model_entry["method"])].iloc[0]
            if any(not np.isclose(value,measured[key],atol=1e-12) for key,value in reconstructed.items()):
                raise ValueError("Saved metrics cannot be reconstructed")
            checks.append({"seed":model_entry["seed"],"method":model_entry["method"],"hospital":hospital,
                           "predictions_and_metrics_reconstructed":True})
    exp.dump_json(OUT/"report/validation.json",{"saved_model_prediction_metric_checks":checks,
        "all27_heads_precede_bank_freeze":True,"head_count":27,"target_records":54,
        "source_bank_sha256":exp.sha256(OUT/"fits/source_heads_frozen.json"),
        "prediction_file_sha256":exp.sha256(OUT/"evaluation/predictions.csv")})
    return manifest,metrics,predictions


def build_bootstrap(predictions):
    rng=np.random.default_rng(2027)
    intervals,contrasts,descriptive=[],[],[]
    arrays={}
    hospital_valid={}
    methods=list(LABELS)
    attempt_records=[]
    for hospital in exp.TARGETS:
        h=predictions[predictions["hospital"]==hospital]
        patient_order=sorted(h["patient"].unique())
        draws=rng.integers(0,len(patient_order),size=(ATTEMPTS,len(patient_order)))
        counts=np.stack([np.bincount(draw,minlength=len(patient_order)) for draw in draws])
        first=h[(h["method"]==methods[0])&(h["seed"]==exp.SEEDS[0])]
        patient_counts=first.groupby("patient")["label"].agg(["sum","count"]).reindex(patient_order)
        p=counts@patient_counts["sum"].to_numpy()
        negative=counts@(patient_counts["count"]-patient_counts["sum"]).to_numpy()
        valid=(p>0)&(negative>0)
        hospital_valid[hospital]=valid
        attempt_records.append({"hospital":hospital,"attempts":ATTEMPTS,"valid_class_complete":int(valid.sum()),
                                "undefined_class_incomplete":int((~valid).sum()),"patient_count":len(patient_order)})
        for method in methods:
            result=bootstrap_method(h[h["method"]==method],counts,patient_order)
            arrays[(hospital,method)]=result
            for metric,values in result.items():
                good=valid&np.isfinite(values)
                if not np.any(good):
                    raise ValueError("No class-complete bootstrap attempts")
                low,high=np.quantile(values[good],[.025,.975])
                intervals.append({"hospital":hospital,"method":method,"metric":metric,
                    "ci_low":float(low),"ci_high":float(high),"valid_draws":int(good.sum()),
                    "undefined_draws":int((~good).sum()),"attempts":ATTEMPTS,
                    "resampling_unit":"patient","seed":2027,
                    "estimand":"mean metric across fixed three head seeds; observed-hospital patient uncertainty"})
            print(f"[BOOTSTRAP] hospital{hospital} {method} complete",flush=True)
        for baseline in ["linear_logistic","single_centroid_group_weighted","multi_prototype_unbalanced"]:
            delta=arrays[(hospital,"cbmp")]["balanced_accuracy"]-arrays[(hospital,baseline)]["balanced_accuracy"]
            good=valid&np.isfinite(delta)
            lo,hi=np.quantile(delta[good],[.025,.975])
            contrasts.append({"hospital":hospital,"contrast":f"cbmp minus {baseline}","metric":"balanced_accuracy",
                              "ci_low":float(lo),"ci_high":float(hi),"valid_draws":int(good.sum())})
    # These summaries describe only two observed hospitals; they are not new
    # draws of hospitals or uncertainty for an unseen hospital population.
    for method in methods:
        for metric in METRICS:
            pair=np.column_stack([arrays[(hospital,method)][metric] for hospital in exp.TARGETS])
            good=np.all(np.isfinite(pair),axis=1)&np.logical_and.reduce([hospital_valid[h] for h in exp.TARGETS])
            for scope,values in [("descriptive_mean_two_observed_hospitals",pair.mean(axis=1)),
                                 ("descriptive_min_two_observed_hospitals",pair.min(axis=1))]:
                lo,hi=np.quantile(values[good],[.025,.975])
                descriptive.append({"scope":scope,"method":method,"metric":metric,"ci_low":float(lo),"ci_high":float(hi),
                                    "valid_draws":int(good.sum()),"generalizes_hospital_sampling_uncertainty":False})
    exp.write_csv(OUT/"report/patient_bootstrap_intervals.csv",intervals)
    exp.write_csv(OUT/"report/paired_cbmp_contrasts.csv",contrasts)
    exp.write_csv(OUT/"report/descriptive_two_hospital_intervals.csv",descriptive)
    exp.dump_json(OUT/"report/bootstrap_protocol.json",{"attempts_per_hospital":ATTEMPTS,"random_seed":2027,
        "patient_draws_shared_between_methods_and_head_seeds":True,"trained_heads_fixed":True,
        "patient_counts_and_undefined_draws":attempt_records,"percentiles":[.025,.975],
        "hospital_population_inference":False,"seed_variation_separate_from_patient_CI":True})
    return pd.DataFrame(intervals),pd.DataFrame(contrasts)


def make_figures(summary,intervals,metrics):
    order=list(LABELS)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True)
    for axis,hospital in zip(axes,exp.TARGETS):
        rows=summary[summary["hospital"]==hospital].set_index("method").loc[order]
        ci=intervals[(intervals["hospital"]==hospital)&(intervals["metric"]=="balanced_accuracy")].set_index("method").loc[order]
        values=rows["balanced_accuracy_mean"].to_numpy()*100
        low,high=ci["ci_low"].to_numpy()*100,ci["ci_high"].to_numpy()*100
        for i,method in enumerate(order):
            color="#c34853" if method=="cbmp" else "#24748e"
            axis.plot([low[i],high[i]],[i,i],color=color,lw=2)
            axis.scatter(values[i],i,color=color,s=45,zorder=3)
        axis.set_yticks(range(len(order)),[LABELS[m] for m in order])
        axis.set_xlim(0,100)
        axis.grid(axis="x",alpha=.2)
        axis.axvline(50,color="gray",ls="--",lw=.9)
        axis.set_xlabel("Balanced accuracy (%)")
        axis.set_title(f"Hospital {hospital}: {'primary test' if hospital==2 else 'secondary test'}")
    axes[0].invert_yaxis()
    fig.suptitle("Independent CAMELYON17-WILDS transfer subset\nFixed ResNet18; patient bootstrap 95% intervals; mean over three fitting seeds",fontsize=12)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(exp.guarded(OUT/f"report/figures/hospital_balanced_accuracy.{ext}"),dpi=200,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharey=True)
    for axis,hospital in zip(axes,exp.TARGETS):
        rows=metrics[metrics["hospital"]==hospital]
        for i,m in enumerate(order):
            values=rows[rows["method"]==m]["auroc"].to_numpy()*100
            axis.scatter(values,np.full(len(values),i),s=26,color="#c34853" if m=="cbmp" else "#24748e",alpha=.7)
        axis.set_yticks(range(len(order)),[LABELS[m] for m in order])
        axis.set_xlim(0,100)
        axis.set_xlabel("AUROC (%) across fitting seeds 17, 29, 43")
        axis.set_title(f"Hospital {hospital}")
        axis.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Fixed encoder; source-only model selection")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(exp.guarded(OUT/f"report/figures/hospital_auroc_seeds.{ext}"),dpi=200,bbox_inches="tight")
    plt.close(fig)


def report():
    exp.verify_protocol()
    snapshot=OUT/"run_source_snapshot.py"
    if not snapshot.exists() or exp.sha256(snapshot)!=exp.sha256(ROOT/"exp5_camelyon.py"):
        raise ValueError("Executed experiment source snapshot differs from final source")
    if not (OUT/"evaluation/complete.json").exists():
        raise FileNotFoundError("Actual54 evaluations must finish before report generation")
    manifest,metrics,predictions=validate_predictions()
    intervals,contrasts=build_bootstrap(predictions)
    grouped=metrics.groupby(["hospital","method"],sort=False)
    summary=grouped[list(METRICS)].agg(["mean","std"]).reset_index()
    summary.columns=["_".join([part for part in col if part]) if isinstance(col,tuple) else col for col in summary.columns]
    exp.write_csv(OUT/"report/summary_by_hospital.csv",summary.to_dict("records"))
    records=[]
    for method in LABELS:
        sub=metrics[metrics["method"]==method]
        perseed=sub.groupby("seed")["balanced_accuracy"].agg(["mean","min"])
        hospital_means=sub.groupby("hospital")["balanced_accuracy"].mean()
        records.append({"method":method,"descriptive_mean_two_hospitals":float(perseed["mean"].mean()),
                        "descriptive_worst_two_hospitals":float(hospital_means.min()),
                        "seed_sd_mean":float(perseed["mean"].std()),
                        "seed_sd_of_per_seed_hospital_minima":float(perseed["min"].std()),
                        "descriptive_worst_estimand":"minimum of hospital-specific mean balanced accuracies across fitting seeds"})
    exp.write_csv(OUT/"report/descriptive_two_hospital_summary.csv",records)
    make_figures(summary,intervals,metrics)
    tex=[r"% Custom frozen-transfer subset; not an official WILDS leaderboard result.",r"\begin{table}[t]",r"\centering",
         r"\caption{Independent hospital transfer on a prespecified patient-capped CAMELYON17-WILDS subset. Frozen ImageNet ResNet18; source-only selection. Entries are balanced accuracy (\%) averaged over three fitting seeds. Hospital 2 is the primary test; hospital 1 is a secondary unseen hospital.}",
         r"\label{tab:camelyon_transfer}",r"\begin{tabular}{lrr}",r"\hline",r"Method & Hospital 2 & Hospital 1 \\",r"\hline"]
    indexed=summary.set_index(["hospital","method"])
    for method,label in LABELS.items():
        tex.append(f"{label} & {indexed.loc[(2,method),'balanced_accuracy_mean']*100:.1f} & {indexed.loc[(1,method),'balanced_accuracy_mean']*100:.1f}" + r" \\")
    tex += [r"\hline",r"\end{tabular}",r"\end{table}"]
    exp.guarded(OUT/"report/table_camelyon.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    audit=json.loads((OUT/"audit/metadata_complete.json").read_text())
    cohort=manifest.groupby(["partition","center"]).agg(patches=("sample_id","size"),patients=("patient","nunique"),positive=("label","sum")).reset_index()
    text=["# Independent CAMELYON17-WILDS frozen-transfer subset","",
        "Actual experiment completed with 27 source-trained heads and 54 target records. All 27 heads were saved and hashed before either target hospital was scored.","",
        "The protocol was fixed before this dataset's target scores. The [community mirror](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/tree/"+exp.PIN+") is pinned at `"+exp.PIN+"`; all 21 full shards match their published LFS SHA256. It is not an author-official redistribution. Frozen ImageNet ResNet18 and the capped subset make this a custom transfer experiment, not a full WILDS benchmark or [official leaderboard submission](https://wilds.stanford.edu/submit/).","",
        "Training uses only original train patches from hospitals 0, 3, 4. All tuning and temperature calibration use inner leave-one-source-hospital-out validation. Original source ID-validation patches are excluded from image extraction and fitting. Hospital 2 is the primary independent test; hospital 1 is a secondary unseen hospital. Neither selects settings.","",
        "This task classifies tumor presence in lymph-node histology patches, with labels defined on the central 32-by-32 region of each 96-by-96 image. Heads are fitted anew using this dataset's source hospitals. This tests replication of the classifier mechanism on an independent task; it is not zero-shot transfer of PolypGen-trained classifiers.","",
        f"Label-blind SHA256 ranking retains at most 400 patches per patient within each eligible partition, preserving all {audit['selected_patients']} available eligible patients and {audit['selected_patches']:,} selected patches. No replacement was chosen using class labels or scores. Equal patient caps can change class proportions relative to the original patch-balanced dataset.","",
        "## Cohort","","|Partition|Hospital|Patches|Patients|Tumor patches|","|---|---:|---:|---:|---:|"]
    for row in cohort.to_dict("records"):
        text.append(f"|{row['partition']}|{row['center']}|{row['patches']}|{row['patients']}|{row['positive']}|")
    text += ["","## Primary and secondary results","","Balanced accuracy is mean across fitting seeds 17, 29, 43. Parentheses give whole-patient bootstrap 95% intervals, using 2,000 attempts with trained heads fixed. These intervals reflect patients within the observed hospital; they do not represent uncertainty across new hospitals.","",
             "|Method|Primary hospital 2 BA|Secondary hospital 1 BA|Primary AUROC|Secondary AUROC|","|---|---:|---:|---:|---:|"]
    ci=intervals[intervals["metric"]=="balanced_accuracy"].set_index(["hospital","method"])
    for method,label in LABELS.items():
        cells=[]
        for hospital in exp.TARGETS:
            value=indexed.loc[(hospital,method),"balanced_accuracy_mean"]*100
            lo,hi=ci.loc[(hospital,method),["ci_low","ci_high"]]*100
            cells.append(f"{value:.1f} ({lo:.1f}--{hi:.1f})")
        auc=[indexed.loc[(h,method),"auroc_mean"]*100 for h in exp.TARGETS]
        text.append(f"|{label}|{cells[0]}|{cells[1]}|{auc[0]:.1f}|{auc[1]:.1f}|")
    cbmp=indexed.loc[(2,"cbmp"),"balanced_accuracy_mean"]
    linear=indexed.loc[(2,"linear_logistic"),"balanced_accuracy_mean"]
    text += ["",f"Primary test CBMP minus linear logistic balanced accuracy: {(cbmp-linear)*100:+.2f} percentage points. Interpret this fixed comparison with the paired patient-bootstrap intervals in `paired_cbmp_contrasts.csv`. No settings were changed after observing it.","",
        "## Validation and limits","",
        "- Per-file ClassLabel hospital and label names were decoded and checked against patient//20 and slide//10, using the [author's metadata generation logic](https://github.com/p-lambda/wilds/blob/main/dataset_preprocessing/camelyon17/generate_final_metadata.py) and [official dataset loader](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py). Coordinate identities and original image IDs were audited.",
        "- Patient groups, source folds, class coverage and exact content duplicates were checked. Cross-hospital duplicated content, contradictory labels or duplicate-linked patients would stop the experiment.",
        "- Every saved model reconstructs its target predictions, calibrated probabilities and metrics. The source freeze manifest predates all target scoring.",
        "- AUROC, macro F1, sensitivity, specificity, NLL and Brier score, three-seed variability and patient bootstrap intervals are available in the accompanying CSV files.",
        "- Only two target hospitals are observed. Descriptive means and minima across them are not five-hospital LOCO results or evidence of general robustness to every hospital.",
        "- The bootstrap holds selected patients' patch subsamples, trained heads and encoder fixed; it does not include full model-selection or patch-subsampling uncertainty.","",
        "## Artifacts","","- `../audit/`: full original metadata, selected label-blind cohort, per-file schemas and image/content hashes.",
        "- `../fits/`: raw source candidate searches, all 27 final model objects and pre-target source freeze manifest.",
        "- `../evaluation/`: all 54 metric records and per-image predictions.",
        "- `summary_by_hospital.csv`: mean and standard deviation across fitting seeds.",
        "- `patient_bootstrap_intervals.csv`, `paired_cbmp_contrasts.csv`: fixed-head within-hospital patient uncertainty.",
        "- `figures/`: PNG and vector PDF figures; `table_camelyon.tex`: standalone Overleaf table.",""]
    exp.guarded(OUT/"report/RESULTS.md").write_text("\n".join(text),encoding="utf-8")
    exp.dump_json(OUT/"completion.json",{"complete":True,"source_models":27,"target_evaluations":54,
        "patches":len(manifest),"patients":int(manifest["patient"].nunique()),"protocol_sha256":exp.sha256(ROOT/"PROTOCOL.md"),
        "acquisition_revision":exp.PIN,"prediction_sha256":exp.sha256(OUT/"evaluation/predictions.csv"),
        "report_sha256":exp.sha256(OUT/"report/RESULTS.md"),"bootstrap_attempts":ATTEMPTS,
        "executed_source_snapshot_sha256":exp.sha256(snapshot),
        "software_versions":{name:importlib.metadata.version(name) for name in
            ["numpy","pandas","scipy","scikit-learn","torch","torchvision","pyarrow","Pillow","joblib","matplotlib"]},
        "python_version":sys.version,
        "source_script_sha256":exp.sha256(ROOT/"exp5_camelyon.py"),"report_script_sha256":exp.sha256(__file__)})
    exp.guarded(OUT/"COMPLETE").write_text("Independent subset:27 frozen source heads,54 target evaluations,validated report.\n",encoding="utf-8")
    print("[EXP5] REPORT COMPLETE",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        report()
