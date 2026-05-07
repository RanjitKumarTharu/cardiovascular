from pydantic import BaseModel

class Cardio(BaseModel):
    age: float
    height: int
    weight: int
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int