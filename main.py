from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hola, este es nuestro Hub de Conocimiento I+D"}
@app.get("/buscar")
def buscar(q: str):
    # Lógica de búsqueda (inicialmente un placeholder)
    return {"resultado": f"Aquí irían los resultados para: {q}"}
