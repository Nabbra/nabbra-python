from enum import Enum

BASE_URL = "https://nabbra.frevva.com/api/v1"

class Endpoints(Enum):
    TOKEN_VALIDATOR = f"{BASE_URL}/profile"

    AUDIOGRAM_SAVE = f"{BASE_URL}/audiograms"
    AUDIOGRAMS = f"{BASE_URL}/audiograms"