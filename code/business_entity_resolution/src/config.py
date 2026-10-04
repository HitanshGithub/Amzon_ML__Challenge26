"""Shared paths and hyper-parameters. Everything can be overridden from the CLI (see run_pipeline.py)."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    data_dir: str = "dataset"          # contains train/ and test/
    work_dir: str = "work"             # intermediate parquet files, models
    out_dir: str = "output"            # final submission files
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)

    # --- blocking ---
    joint_w: float = 0.4               # weight of name vs address in the joint TF-IDF vector
    k_joint: int = 40                  # top-k per S1 entity on the joint name+address score
    k_name_extra: int = 10             # + name-only top-k (helps records with an empty address)
    name_max_df: float = 0.05          # drop char 3-grams present in more than this share of names
    name_min_sim: float = 0.10
    query_chunk: int = 200_000         # S1 rows per sparse top-n multiplication
    block_countries: str = ""          # if set, re-block only these countries and merge
    rescue_k: int = 20                 # name-only top-k among EMPTY-address pool records (0 = off)
    rescue_min_sim: float = 0.3

    # --- training ---
    train_s1_frac: float = 0.35        # fraction of train S1 entities used to fit the model
    val_s1_frac: float = 0.10          # fraction held out for threshold tuning / scoring
    # Test has ~2.3 unmatched S2/S3 records per S1 entity vs ~1.2 in train (60% vs 74% of the pool
    # is matched). Removing this share of train S1 entities (never trained/validated on) turns their
    # true records into unowned distractors, reproducing the test distractor density.
    drop_s1_frac: float = 0.0          # v5: off - the test's extra decoys are unrelated businesses
                                       # (empty-address rate 2.67% = 4.4% x 60% matched), not copies
                                       # of removed entities; the drop made the model doubt
                                       # empty-address copies (68% of remaining recall loss)
    seed: int = 42
    lgb_rounds: int = 1500
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
    })

    def path(self, *parts):
        p = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
