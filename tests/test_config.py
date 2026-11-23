from pathlib import Path

from src import config


def test_diretorios_do_projeto_existem():
    for d in (config.DATA_RAW, config.DATA_INTERIM, config.DATA_PROCESSED,
              config.MODELS_DIR, config.OUTPUT_PREDICTIONS, config.OUTPUT_REPORTS):
        assert isinstance(d, Path) and d.is_dir(), f"Diretório ausente: {d}"


def test_classes_binarias_coerentes():
    assert config.CLASS_NAMES[config.CLASS_MEDIO] == "Médio"
    assert config.CLASS_NAMES[config.CLASS_GRANDE] == "Grande"
    assert config.CLASS_MEDIO == 0 and config.CLASS_GRANDE == 1


def test_campos_obrigatorios_tem_alias():
    for field in config.REQUIRED_FIELDS:
        assert field in config.COLUMN_ALIASES
    for group in config.REQUIRED_ONE_OF:
        assert all(f in config.COLUMN_ALIASES for f in group)


def test_rotulo_e_identificadores_nunca_sao_features():
    for field in ("porte_obs", "porte_oficial", "cnpj", "cnpj_raiz"):
        assert field in config.NEVER_FEATURES
