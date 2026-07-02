"""
check_data/test_data.py — Etapa 4: Tests de calidad del dataset de fraude.

Validaciones específicas al dominio de detección de fraude:
  - Presencia de al menos N fraudes en cada split (necesario para métricas)
  - Features V1-V28 en rango [-50, 50] (PCA-transformadas)
  - Amount_log normalizado en [0, 1]
  - Time_sin y Time_cos en [-1, 1]
  - No duplicados exactos
  - Consistencia de tasa de fraude entre train y test

Ejecutar: pytest check_data/test_data.py -v
"""
import pandas as pd
import numpy as np
import pytest

V_COLS = [f"V{i}" for i in range(1, 29)]
MIN_FRAUDES_POR_SPLIT = 3   # mínimo estadístico para calcular métricas


class TestEstructura:
    def test_columnas_presentes_train(self, df_train):
        expected = V_COLS + ["Amount_log", "Time_sin", "Time_cos", "Class"]
        missing = set(expected) - set(df_train.columns)
        assert not missing, f"Columnas faltantes en train: {missing}"

    def test_columnas_presentes_test(self, df_test):
        expected = V_COLS + ["Amount_log", "Time_sin", "Time_cos", "Class"]
        missing = set(expected) - set(df_test.columns)
        assert not missing, f"Columnas faltantes en test: {missing}"

    def test_sin_nulos_train(self, df_train):
        nulos = df_train.isnull().sum().sum()
        assert nulos == 0, f"Hay {nulos} nulos en el dataset de entrenamiento"

    def test_sin_nulos_test(self, df_test):
        nulos = df_test.isnull().sum().sum()
        assert nulos == 0, f"Hay {nulos} nulos en el dataset de test"

    def test_target_binario(self, df_train):
        valores = set(df_train["Class"].unique())
        assert valores <= {0, 1}, f"Target tiene valores inesperados: {valores}"


class TestFraude:
    def test_fraudes_suficientes_en_train(self, df_train):
        n_fraudes = df_train["Class"].sum()
        assert n_fraudes >= MIN_FRAUDES_POR_SPLIT, (
            f"Solo {n_fraudes} fraudes en train — insuficiente para métricas. "
            f"Mínimo requerido: {MIN_FRAUDES_POR_SPLIT}"
        )

    def test_fraudes_suficientes_en_test(self, df_test):
        n_fraudes = df_test["Class"].sum()
        assert n_fraudes >= MIN_FRAUDES_POR_SPLIT, (
            f"Solo {n_fraudes} fraudes en test — insuficiente para métricas. "
            f"Mínimo requerido: {MIN_FRAUDES_POR_SPLIT}"
        )

    def test_tasa_fraude_consistente(self, df_train, df_test):
        tasa_train = df_train["Class"].mean()
        tasa_test  = df_test["Class"].mean()
        diff = abs(tasa_train - tasa_test)
        assert diff < 0.01, (
            f"Tasa de fraude muy distinta: train={tasa_train:.4f}, test={tasa_test:.4f}. "
            f"Diferencia: {diff:.4f} (max permitido: 0.01). "
            f"Verifica el stratify en segregate/run.py"
        )

    def test_no_data_leakage_filas(self, df_train, df_test):
        """Verifica que no haya filas duplicadas entre train y test (data leakage)."""
        key_cols = [f"V{i}" for i in range(1, 4)]  # primeras 3 componentes PCA como proxy
        key_cols = [c for c in key_cols if c in df_train.columns and c in df_test.columns]
        if not key_cols:
            return
        train_keys = set(df_train[key_cols].round(6).apply(tuple, axis=1))
        test_keys  = set(df_test[key_cols].round(6).apply(tuple, axis=1))
        overlap = train_keys & test_keys
        pct_overlap = len(overlap) / len(test_keys) if test_keys else 0
        assert pct_overlap < 0.01, (
            f"Posible data leakage: {len(overlap)} filas ({pct_overlap:.1%}) en común entre train y test"
        )


class TestRangos:
    def test_v_features_rango(self, df_train):
        for col in V_COLS:
            if col in df_train.columns:
                max_val = df_train[col].abs().max()
                assert max_val <= 50, f"'{col}' tiene valor absoluto > 50: {max_val:.2f}"

    def test_amount_log_normalizado(self, df_train):
        assert df_train["Amount_log"].min() >= -0.01, (
            f"Amount_log tiene valores negativos: min={df_train['Amount_log'].min():.4f}"
        )
        assert df_train["Amount_log"].max() <= 1.01, (
            f"Amount_log > 1: max={df_train['Amount_log'].max():.4f}"
        )

    def test_time_ciclico_rango(self, df_train):
        for col in ["Time_sin", "Time_cos"]:
            if col in df_train.columns:
                assert df_train[col].min() >= -1.01, f"{col} < -1: {df_train[col].min():.4f}"
                assert df_train[col].max() <= 1.01,  f"{col} > 1: {df_train[col].max():.4f}"

    def test_sin_duplicados_exactos(self, df_train):
        n_dup = df_train.duplicated().sum()
        assert n_dup == 0, f"Hay {n_dup} filas duplicadas exactas en train"

    def test_tamano_minimo(self, df_train):
        assert len(df_train) >= 100, f"Dataset de entrenamiento muy pequeño: {len(df_train)} filas"
