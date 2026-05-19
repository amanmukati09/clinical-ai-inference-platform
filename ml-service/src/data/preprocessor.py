"""
Clinical data preprocessor for diabetic readmission prediction.
Handles the UCI Diabetes 130-US Hospitals dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Columns to drop — too much missing data to be useful
HIGH_MISSING_COLS = [
    'weight', 'max_glu_serum', 'A1Cresult',
    'payer_code', 'medical_specialty'
]

# IDs — not features, just identifiers
ID_COLS = ['encounter_id', 'patient_nbr']

# Medication columns — these are ordinal: No, Steady, Up, Down
MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

MEDICATION_MAP = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}

# Age is a range string — map to midpoint
AGE_MAP = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}


def load_raw_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary target: 1 if readmitted within 30 days, 0 otherwise.
    Clinical rationale: <30 day readmission = CMS penalty event,
    indicates care transition failure at discharge.
    """
    df = df.copy()
    df['target'] = (df['readmitted'] == '<30').astype(int)
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace sentinel missing value
    df = df.replace('?', np.nan)

    # Drop high-missing and irrelevant columns
    cols_to_drop = HIGH_MISSING_COLS + ID_COLS + ['readmitted']
    df = df.drop(columns=cols_to_drop)
    logger.info(f"After dropping columns: {df.shape[1]} features remaining")

    # Remove invalid gender entries
    df = df[df['gender'] != 'Unknown/Invalid']

    # Remove encounters where patient was discharged to hospice or died
    # These patients cannot be readmitted — they would corrupt the label
    # Discharge disposition 11=Expired, 13=Hospice, 14=Hospice
    df = df[~df['discharge_disposition_id'].isin([11, 13, 14])]
    logger.info(f"After removing expired/hospice: {df.shape[0]} rows")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age: string range -> numeric midpoint
    df['age'] = df['age'].map(AGE_MAP)

    # Medications: ordinal encoding
    for col in MEDICATION_COLS:
        if col in df.columns:
            df[col] = df[col].map(MEDICATION_MAP).fillna(0)

    # Diagnosis codes: ICD-9 -> broad clinical category
    # This reduces cardinality from thousands of codes to 9 categories
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df.columns:
            df[diag_col] = df[diag_col].fillna('0').apply(map_diag_to_category)

    # Binary encodings
    df['change'] = (df['change'] == 'Ch').astype(int)
    df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)
    df['gender'] = (df['gender'] == 'Male').astype(int)

    # Race: fill missing with mode, then encode
    df['race'] = df['race'].fillna(df['race'].mode()[0])

    return df


def map_diag_to_category(code: str) -> int:
    """
    Map ICD-9 diagnosis code to broad clinical category.
    Reduces thousands of codes to 9 interpretable groups.
    This is standard practice in clinical ML with ICD codes.
    """
    try:
        if code.startswith('V') or code.startswith('E'):
            return 0
        c = float(code)
        if 390 <= c <= 459 or c == 785: return 1   # Circulatory
        if 460 <= c <= 519 or c == 786: return 2   # Respiratory
        if 520 <= c <= 579 or c == 787: return 3   # Digestive
        if 250 <= c <= 250.99: return 4             # Diabetes
        if 800 <= c <= 999: return 5                # Injury
        if 710 <= c <= 739: return 6                # Musculoskeletal
        if 580 <= c <= 629 or c == 788: return 7   # Genitourinary
        if 140 <= c <= 239: return 8                # Neoplasms
        return 0
    except (ValueError, AttributeError):
        return 0


def encode_categoricals(df: pd.DataFrame, fit: bool = True,
                         encoder_path: str = None) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    logger.info(f"Encoding categorical columns: {categorical_cols}")

    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        if encoder_path:
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoders, f)
            logger.info(f"Saved encoders to {encoder_path}")
    else:
        if encoder_path is None:
            raise ValueError("encoder_path required when fit=False")
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        for col in categorical_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))

    return df


def split_by_patient(df: pd.DataFrame, patient_ids: pd.Series,
                      test_size: float = 0.2, random_state: int = 42):
    """
    Split by patient, not by row.
    Prevents data leakage where the same patient appears in
    both train and test sets.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=patient_ids))
    return df.iloc[train_idx], df.iloc[test_idx]


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    scaler_path: str = None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if scaler_path:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
    return X_train_scaled, X_test_scaled, scaler


def run_pipeline(raw_path: str, output_dir: str):
    """
    Full preprocessing pipeline.
    Saves processed train/test splits and artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    artifact_dir = os.path.join(output_dir, 'artifacts')
    os.makedirs(artifact_dir, exist_ok=True)

    # Load
    df = load_raw_data(raw_path)

    # Save patient IDs before cleaning (needed for group split)
    patient_ids = df['patient_nbr'].copy()

    # Build target
    df = create_target(df)

    # Clean
    df = clean_data(df)

    # Realign patient_ids after row drops
    patient_ids = patient_ids.loc[df.index]

    # Feature engineering
    df = engineer_features(df)

    # Encode
    encoder_path = os.path.join(artifact_dir, 'encoders.pkl')
    df = encode_categoricals(df, fit=True, encoder_path=encoder_path)

    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']

    logger.info(f"Final feature matrix: {X.shape}")
    logger.info(f"Feature columns: {X.columns.tolist()}")

    # Split by patient
    X_train, X_test = split_by_patient(X, patient_ids)
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    logger.info(f"Train positive rate: {y_train.mean():.3f}")
    logger.info(f"Test positive rate: {y_test.mean():.3f}")

    # Scale
    scaler_path = os.path.join(artifact_dir, 'scaler.pkl')
    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train, X_test, scaler_path=scaler_path
    )

    # Save processed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)

    # Save feature names for model interpretability
    feature_names = X_train.columns.tolist()
    with open(os.path.join(artifact_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    logger.info("Pipeline complete. Artifacts saved.")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


if __name__ == '__main__':
    run_pipeline(
        raw_path='/home/aman/clinical-ai-platform/data/raw/diabetic_data.csv',
        output_dir='/home/aman/clinical-ai-platform/data/processed'
    )
