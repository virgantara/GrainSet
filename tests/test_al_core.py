"""Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from al.data import budgets, complement, UnlabeledDataset
from al.acquisition import scores, select_indices
from al.purity import bag_metrics


class TestAlCore(unittest.TestCase):
    def test_budget(self):
        b = budgets(24767,2,50,2)
        self.assertEqual(len(b),25)
        self.assertEqual([b[0][1],b[-1][1]], [495, 12384])
        self.assertTrue(all(a[1]<z[1] for a,z in zip(b,b[1:])))

    def test_complement(self):
        self.assertEqual(complement(6,[0,2,4]).tolist(),[1,3,5])

    def test_uncertainty(self):
        p = np.array([[.9,.1],[.5,.5],[.6,.4]])
        for method in ('least_confidence','entropy','margin'):
            self.assertEqual(np.argmax(scores(p,method)),1)

    def test_selection_no_repeats(self):
        rng = np.random.default_rng(4)
        pool = np.arange(100,200)
        p = rng.dirichlet(np.ones(8),size=100)
        emb = rng.normal(size=(100,512)).astype('float32')
        for m in ('random','least_confidence','entropy','margin','hybrid'):
            q = select_indices(m,pool,20,p,emb,seed=42)
            self.assertEqual(len(set(q)),20)
            self.assertTrue(set(q).issubset(set(pool)))
            np.testing.assert_array_equal(q,select_indices(m,pool,20,p,emb,seed=42))

    def test_uncertainty_dataset_has_no_labels(self):
        import inspect
        self.assertNotIn('label', inspect.getsource(UnlabeledDataset.__getitem__))

    def test_purity_perfect_model(self):
        y = np.array([0]*600+[7]*150)
        p = np.eye(8)[y]
        res = bag_metrics(y,p,repetitions=2,bag_sizes=(100,200),rates=(.01,.05))
        self.assertAlmostEqual(res['clean_mape_soft'],0,places=9)
        self.assertAlmostEqual(res['clean_mape_hard'],0,places=9)
        self.assertGreater(res['always_clean_mape'],0)


if __name__=='__main__': unittest.main()
