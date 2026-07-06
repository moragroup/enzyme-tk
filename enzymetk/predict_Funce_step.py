"""
Funce step: ensemble activity prediction for enzyme-reaction pairs.

Ports Phases 2-4 of the standalone ``predict_ml_ensemble.py`` CLI into a
composable enzyme-tk ``Step``. ``Funce`` is *prediction-only*: the input
DataFrame must already carry the embedding columns (``esm3_mean``, ``rxnfp``,
``substrate_unimol_repr``, ``product_unimol_repr``). Encoding the reaction /
protein stays the caller's job via existing steps (``UniMol``, ``RxnFP``,
``EmbedESM3``).

The step loads an ensemble of 4 attention models (one per EC level) that each
score a pair, then averages predictions and inverse-transforms the enzyme /
reaction feature heads.
"""
from enzymetk.step import Step
import pandas as pd
import numpy as np
import os
import sys
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default location of the (gitignored) model folder shipped alongside the repo.
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / 'data' / 'Funce' / 'models'

# Feature heads the ensemble regresses, in the order the models emit them.
ENZYME_COLS = ['Length', 'Mass', 'Polarity', 'temperature']
REACTION_COLS = [
    'substrates_MolWt', 'substrates_MolLogP',
    'substrates_MaxPartialCharge', 'substrates_MinPartialCharge',
    'products_MolWt', 'products_TPSA', 'products_MolLogP',
    'products_MaxPartialCharge', 'products_MinPartialCharge',
]


def _register_model_classes():
    """Make the pickled ``nn.Module`` classes importable for ``torch.load``.

    The checkpoints pickle live model instances, so the classes must be
    resolvable under whatever module path they were pickled with. We register
    the vendored module under every alias the training code may have used
    (``__main__`` for notebook runs, ``run_ml_04092025``/``colab_ml_train``
    for script runs) and expose the classes on ``__main__`` directly.
    """
    from enzymetk import Funce_model
    import __main__

    __main__.NeuralNetworkModelWithAttention = Funce_model.NeuralNetworkModelWithAttention
    __main__.CrossAttention = Funce_model.CrossAttention

    for alias in ('run_ml_04092025', 'colab_ml_train'):
        sys.modules.setdefault(alias, Funce_model)


def _load_checkpoint(config_path, checkpoint_path):
    """Load a model checkpoint and its configuration (CPU-safe)."""
    import torch
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location=torch.device('cpu')
    )
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    with open(config_path, 'rb') as fh:
        config = pickle.load(fh)
    return model, config, optimizer


def _load_models(model_dir, reaction_level, protein_level, model_type, num_pairs):
    """Load ensemble of models for EC levels 1-4."""
    label = 'run'
    model_idx = 1
    models = []
    enzyme_feature_scaler = None
    reaction_feature_scaler = None
    missing = []

    logger.info(f'Loading Funce models from: {model_dir}')

    for ec_level in range(1, 5):
        base_name = f'{label}_{reaction_level}_{protein_level}_{model_type}_{ec_level}_model_{model_idx}_{num_pairs}'
        config_path = os.path.join(model_dir, f'{base_name}_conf.pkl')
        checkpoint_path = os.path.join(model_dir, f'{base_name}_checkpoint.pth')

        if os.path.exists(config_path) and os.path.exists(checkpoint_path):
            model, config, _ = _load_checkpoint(config_path, checkpoint_path)
            enzyme_feature_scaler = config['enzyme_feature_scaler']
            reaction_feature_scaler = config['reaction_feature_scaler']
            models.append(model)
            logger.info(f'  Loaded EC level {ec_level} model: {base_name}')
        else:
            logger.warning(f'  Model not found for EC level {ec_level}: {config_path}')
            missing.append(config_path)

    if not models:
        raise RuntimeError(
            f'No Funce models were loaded from {model_dir}. '
            f'Check the model_dir and naming '
            f'(run_{reaction_level}_{protein_level}_{model_type}_<ec>_model_1_{num_pairs}_*).'
        )

    logger.info(f'  Loaded {len(models)} models total; {len(missing)} missing')
    return models, enzyme_feature_scaler, reaction_feature_scaler


def retransform_scaled_predictions(test, pred, enzyme_feature_scaler, enzyme_cols,
                                   reaction_feature_scaler, reaction_cols):
    """Inverse-transform scaled predictions back to original scale."""
    pred_enzyme_cols = []
    for i, v in enumerate(enzyme_cols):
        test[f'pred_{v}'] = pred[:, i + 1].detach()
        pred_enzyme_cols.append(f'pred_{v}')

    scaled_cols = enzyme_feature_scaler.inverse_transform(test[pred_enzyme_cols].values)
    for i, v in enumerate(pred_enzyme_cols):
        test[f'inverse_transformed_{v}'] = scaled_cols[:, i]

    pred_reaction_cols = []
    for i, v in enumerate(reaction_cols):
        test[f'pred_{v}'] = pred[:, i + 1 + len(pred_enzyme_cols)].detach()
        pred_reaction_cols.append(f'pred_{v}')

    scaled_cols = reaction_feature_scaler.inverse_transform(test[pred_reaction_cols].values)
    for i, v in enumerate(pred_reaction_cols):
        test[f'inverse_transformed_{v}'] = scaled_cols[:, i]

    test['pred_Activity'] = pred[:, 0].detach()
    return test


def average_retransformed_features(df_to_add, dfs, enzyme_cols, reaction_cols, label=''):
    """Average retransformed features across ensemble models."""
    for col in enzyme_cols + reaction_cols:
        data = np.array([d[f'inverse_transformed_pred_{col}'].values for d in dfs])
        df_to_add[f'{label}{col}_mean'] = np.mean(data, axis=0)
        df_to_add[f'{label}{col}_std'] = np.std(data, axis=0)
    return df_to_add


class Funce(Step):
    """Ensemble activity prediction for enzyme-reaction pairs.

    Expects a pre-encoded DataFrame (embedding columns already present) and
    appends prediction / retransformed-feature columns prefixed with ``name``.
    """

    def __init__(self, id_col: str,
                 protein_emb_col: str = 'esm3_mean',
                 rxn_col: str = 'rxnfp',
                 sub_col: str = 'substrate_unimol_repr',
                 prod_col: str = 'product_unimol_repr',
                 name: str = 'Funce',
                 model_dir: str = None,
                 reaction_level: str = 'easy',
                 protein_level: str = '0-50',
                 model_type: str = 'ESRP',
                 num_pairs: int = 500000,
                 label_col: str = None,
                 num_threads: int = 1,
                 tmp_dir: str = None,
                 env_name: str = 'enzymetk'):
        super().__init__()
        self.conda = env_name
        self.env_name = env_name
        self.id_col = id_col
        self.protein_emb_col = protein_emb_col
        self.rxn_col = rxn_col
        self.sub_col = sub_col
        self.prod_col = prod_col
        self.name = name
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.reaction_level = reaction_level
        self.protein_level = protein_level
        self.model_type = model_type
        self.num_pairs = num_pairs
        self.label_col = label_col
        self.num_threads = num_threads or 1
        self.tmp_dir = tmp_dir
        self.logger = logging.getLogger(__name__)

    def _to_tensor(self, series, device):
        """Build a float32 tensor from a column of (possibly nested) arrays."""
        import torch
        arr = np.array([np.asarray(x).flatten().astype(np.float32)
                        for x in series.values])
        return torch.tensor(arr, dtype=torch.float32).to(device)

    def _predict(self, df, models, enzyme_feature_scaler, reaction_feature_scaler):
        import torch
        from sklearn.metrics import f1_score, average_precision_score

        torch.set_float32_matmul_precision('high')
        torch.set_default_dtype(torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'Funce inference on {len(df)} samples with {len(models)} models (device={device})')

        all_model_preds = []
        validation_dfs = []
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                for model_idx, model in enumerate(models):
                    model.eval()
                    for module in model.modules():
                        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                            module.eval()
                            module.track_running_stats = False
                            module.running_mean = module.running_mean.detach()
                            module.running_var = module.running_var.detach()
                    model.to(device)
                    model.eval()

                    X_enzyme = self._to_tensor(df[self.protein_emb_col], device)
                    X_product = self._to_tensor(df[self.prod_col], device)
                    X_substrate = self._to_tensor(df[self.sub_col], device)
                    X_reaction = self._to_tensor(df[self.rxn_col], device)

                    # NOTE the forward signature order: (enzyme, product, substrate, reaction)
                    pred = model(X_enzyme, X_product, X_substrate, X_reaction)

                    # Sigmoid on the activity logit (normally applied in the loss)
                    pred = pred.cpu()
                    probs = torch.sigmoid(pred[:, 0].squeeze())
                    pred[:, 0] = probs.float()

                    validation_df = retransform_scaled_predictions(
                        df, pred, enzyme_feature_scaler, ENZYME_COLS,
                        reaction_feature_scaler, REACTION_COLS,
                    )
                    all_model_preds.append(pred.numpy())
                    validation_dfs.append(validation_df.copy())
                    self.logger.info(f'  Model {model_idx + 1}/{len(models)} done')

        all_model_preds = np.stack(all_model_preds, axis=0)
        mean_preds = np.mean(all_model_preds, axis=0)
        std_preds = np.std(all_model_preds, axis=0)

        # Optional classification metrics (only if ground truth is provided)
        if self.label_col is not None and self.label_col in df.columns:
            y_true = df[self.label_col].values
            y_pred = np.round(mean_preds[:, 0])
            acc = float(np.mean(y_pred == y_true))
            f1 = f1_score(y_true, y_pred, average='macro')
            auprc = average_precision_score(y_true, y_pred)
            self.logger.info(f'  Accuracy: {acc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}')

        return mean_preds, std_preds, validation_dfs

    def __execute(self, df: pd.DataFrame) -> pd.DataFrame:
        _register_model_classes()
        models, enzyme_feature_scaler, reaction_feature_scaler = _load_models(
            str(self.model_dir), self.reaction_level, self.protein_level,
            self.model_type, self.num_pairs,
        )
        mean_preds, std_preds, ensemble_dfs = self._predict(
            df, models, enzyme_feature_scaler, reaction_feature_scaler,
        )

        df[f'{self.name}_prediction'] = mean_preds[:, 0]
        df[f'{self.name}_std_preds'] = std_preds[:, 0]
        df[f'{self.name}_epistemic'] = True
        df = average_retransformed_features(
            df, ensemble_dfs, ENZYME_COLS, REACTION_COLS, f'{self.name}_',
        )
        return df

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.num_threads > 1:
            df_list = np.array_split(df, self.num_threads)
            # Each chunk reloads the ensemble; single-thread is the primary path.
            return pd.concat([self.__execute(chunk.copy()) for chunk in df_list])
        return self.__execute(df)
