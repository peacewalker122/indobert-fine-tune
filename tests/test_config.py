from src.config import ARTIFACT_VERSION, MODEL_NAME


def test_model_name_is_multilingual_cased_bert():
    assert MODEL_NAME == "google-bert/bert-base-multilingual-cased"


def test_artifact_version_is_multilingual_intent_slot_v2():
    assert ARTIFACT_VERSION == "multilingual-intent-slot-v2"
