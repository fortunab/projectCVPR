import numpy as np
import pytest
from cbmp.models import CBMP, FrozenHead, class_weights, normalize


def test_center_mass_and_frame_replication_invariance():
    y = np.zeros(5, dtype=int)
    c = np.array(['a','a','a','a','b'])
    g = np.array(['a1','a1','a1','a2','b1'])
    w = class_weights(y,c,g,beta=1)
    assert w[c=='a'].sum() == pytest.approx(.5)
    assert w[g=='a1'].sum() == pytest.approx(.25)
    assert w[g=='a2'].sum() == pytest.approx(.25)
    z = normalize([[1,.2],[1,.2],[1,.2],[.2,1],[1,1]])
    m = CBMP(k=1,beta=1).fit(z,y,c,g)
    reduced = [0,3,4]
    n = CBMP(k=1,beta=1).fit(z[reduced],y[reduced],c[reduced],g[reduced])
    np.testing.assert_allclose(m.prototypes_[0],n.prototypes_[0])


def test_single_centroid_and_missing_center_classes():
    x = normalize([[1,0],[0,1],[1,1],[-1,0]])
    y = np.array([0,0,0,1])
    c = np.array(['a','a','b','b'])
    g = np.array(['p1','p2','p3','p4'])
    model = CBMP(k=1,beta=0).fit(x,y,c,g)
    np.testing.assert_allclose(model.prototypes_[0][0],normalize(x[:3].mean(axis=0)[None])[0])
    assert np.isfinite(model.decision_function(x)).all()


def test_multimodality_and_degenerate_directions():
    x = np.array([[1.,0],[-1.,0],[0,1],[0,-1]])
    y = np.array([0,0,1,1])
    c = np.array(['a','b','a','b'])
    g = np.array(['p1','p2','p3','p4'])
    model = CBMP(k=2,seed=3).fit(x,y,c,g)
    np.testing.assert_array_equal(model.predict(x),y)
    for p in model.prototypes_:
        np.testing.assert_allclose(np.linalg.norm(p,axis=1),1)
    cancel = CBMP(k=1).fit(x,y,c,g)
    assert np.isfinite(cancel.decision_function(x)).all()
    duplicate = CBMP(k=8).fit(np.tile([1.,0],(4,1)),y,c,g)
    assert all(len(p)==1 for p in duplicate.prototypes_)


def test_reproducibility_and_invalid_inputs():
    rng = np.random.default_rng(12)
    x=rng.normal(size=(20,4)); y=np.arange(20)%2
    c=np.array(['a','b']*10); g=np.array([str(i) for i in range(20)])
    a=CBMP(k=4,seed=42).fit(x,y,c,g)
    b=CBMP(k=4,seed=42).fit(x,y,c,g)
    np.testing.assert_array_equal(a.decision_function(x),b.decision_function(x))
    with pytest.raises(ValueError): normalize([[0,0]])
    with pytest.raises(ValueError): normalize([[float('nan'),1]])
    with pytest.raises(ValueError): class_weights(y,c,g,beta=-1)


@pytest.mark.parametrize('cosine',[False,True])
def test_matched_heads_learn_separable_data(cosine):
    rng=np.random.default_rng(4)
    x=np.r_[rng.normal([1,0],.1,size=(12,2)),rng.normal([-1,0],.1,size=(12,2))]
    y=np.r_[np.zeros(12,dtype=int),np.ones(12,dtype=int)]
    head=FrozenHead(cosine=cosine).fit(x,y,np.array(['a']*24),np.arange(24).astype(str))
    np.testing.assert_array_equal(head.decision_function(x).argmax(axis=1),y)
