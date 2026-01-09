from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Carregar o Pipeline (que já contém o scaler e o modelo)
pipeline = joblib.load('assets/modelo_classificacao.joblib')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite que o App se conecte
    allow_credentials=True,
    allow_methods=["*"], # Permite POST, GET, etc.
    allow_headers=["*"],
)
# 2. Definir a estrutura exata que o seu modelo espera
class DadosPaciente(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    GenHlth: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    Age: float
    Education: float
    Income: float

@app.post("/predict")
def predict(dados: DadosPaciente):
    # Converte o JSON do App para DataFrame
    df = pd.DataFrame([dados.dict()])
    
    # O pipeline já faz o scaling e a predição sozinho
    probabilidade = pipeline.predict_proba(df)[0][1]
    resultado = 1 if probabilidade > 0.5 else 0
    
    return {
        "diabetes": resultado,
        "probabilidade": round(float(probabilidade), 4)
    }