import csv
import json
import numpy as np
import pytest
from PIL import Image
from cbmp.data import audit_manifest, load_features, load_manifest
from cbmp.cli import main
from cbmp.experiment import inner_splits, run
from cbmp.report import export


def test_nested_splits_never_use_target_and_cache_requires_opt_in(tmp_path):
    cache=tmp_path/'demo.npz'
    main(['demo-features',str(cache)])
    with pytest.raises(ValueError,match='allow-synthetic'): load_features(cache)
    data,_=load_features(cache,True)
    source=np.flatnonzero(data['centers']!='demo_center_0')
    splits,_=inner_splits(data,source,0)
    for train,val in splits:
        assert not set(data['groups'][train]) & set(data['groups'][val])
        assert 'demo_center_0' not in data['centers'][train]
        assert 'demo_center_0' not in data['centers'][val]


def test_end_to_end_artifacts_and_target_labels_do_not_select_params(tmp_path):
    cache=tmp_path/'demo.npz'
    main(['demo-features',str(cache)])
    output=tmp_path/'run'
    results=run(cache,output,['centroid'],[0],True,10)
    assert len(results)==4
    assert json.loads((output/'run.json').read_text())['status']=='complete'
    paper=export(output)
    assert 'SYNTHETIC' in (paper/'summary.tex').read_text()
    assert (paper/'center_performance.pdf').stat().st_size>100
    from cbmp.experiment import select
    data,_=load_features(cache,True)
    source=np.flatnonzero(data['centers']!='demo_center_0')
    splits,_=inner_splits(data,source,0)
    before=select('cbmp',data,splits,0)
    data['y'][data['centers']=='demo_center_0'] = 1-data['y'][data['centers']=='demo_center_0']
    after=select('cbmp',data,splits,0)
    assert before==after


def test_audit_rejects_duplicates_and_missing_group_ids(tmp_path):
    Image.new('RGB',(8,8),'red').save(tmp_path/'a.png')
    Image.new('RGB',(8,8),'red').save(tmp_path/'b.png')
    manifest=tmp_path/'manifest.csv'
    with manifest.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['sample_id','image_path','label','center','group_id'])
        w.writerow(['a','a.png',0,'a','p1']);w.writerow(['b','b.png',1,'b','p2'])
    with pytest.raises(ValueError,match='Audit failed'):audit_manifest(manifest,tmp_path/'audit')
    text=manifest.read_text().replace(',p2',',')
    manifest.write_text(text)
    with pytest.raises(ValueError,match='Blank'):load_manifest(manifest)
