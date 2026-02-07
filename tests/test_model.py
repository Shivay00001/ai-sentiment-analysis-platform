import pytest
from src.core.model import SentimentModel

@pytest.fixture
def model():
    # Mocking or using a tiny model for tests would be better
    # for now we just test the class structure
    return SentimentModel()

def test_model_init(model):
    assert model.model_name is not None
