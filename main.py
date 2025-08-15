from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch, torch.nn as nn

# --- modelo PyTorch (igual ao treino) ---
class RedeSimples(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# --- schemas ---
class Entrada(BaseModel):
    idade: float
    renda: float

class Lote(BaseModel):
    itens: List[Entrada]

# --- api ---
app = FastAPI()
modelo = RedeSimples()
modelo.load_state_dict(torch.load("modelo.pth", map_location="cpu"))
modelo.eval()

# normalização igual ao treino
def norm(idade, renda):
    idade_norm = (idade - 18) / (70 - 18)
    renda_norm = (renda - 1000) / (10000 - 1000)
    return idade_norm, renda_norm

@app.get("/")
def home():
    return {"mensagem": "API online e funcionando"}

@app.post("/predict")
def predict(dados: Entrada):
    i, r = norm(dados.idade, dados.renda)
    x = torch.tensor([[i, r]], dtype=torch.float32)
    return {"risco": float(modelo(x).item())}

@app.post("/predict-batch")
def predict_batch(lote: Lote):
    xs = [norm(item.idade, item.renda) for item in lote.itens]
    x = torch.tensor(xs, dtype=torch.float32)
    with torch.no_grad():
        probs = modelo(x).squeeze(1).tolist()
    return {"riscos": [float(p) for p in probs]}
