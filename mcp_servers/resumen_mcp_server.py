import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mcp.server.fastmcp import FastMCP

model_name = "Qwen/Qwen2.5-1.5B-Instruct"


def _elegir_dispositivo() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        # Algunas GPU (p. ej. la P100 de Kaggle) se detectan pero no son
        # compatibles con las versiones recientes de PyTorch: se prueba una
        # operación real y, si falla, se usa la CPU.
        try:
            (torch.ones(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"GPU no utilizable ({exc}); se usará CPU.", flush=True)
    return "cpu"


device = _elegir_dispositivo()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32 if device == "cpu" else torch.float16,
).to(device)
mcp = FastMCP("ResumenMCP", host="localhost", port=9081)

MAX_TOKENS_FRAGMENTO = 3000

SISTEMA = ("Eres un asistente que resume textos en español de forma fiel, "
           "sin inventar información que no esté en el texto.")

INSTRUCCION = ("Escribe un resumen de este texto en UN SOLO PÁRRAFO de prosa "
               "continua (sin listas, sin viñetas, sin títulos, sin negritas), "
               "de máximo {palabras} palabras. Explica la idea general y menciona "
               "solo los hitos más importantes, redactado con tus propias palabras.")


def _generar(texto: str, palabras: int = 150) -> str:
    mensajes = [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": f'Texto:\n"""\n{texto}\n"""\n\n'
                                    + INSTRUCCION.format(palabras=palabras)},
    ]
    inputs = tokenizer.apply_chat_template(
        mensajes, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        repetition_penalty=1.1,
    )
    nuevos = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(nuevos, skip_special_tokens=True).strip()


def _dividir_en_fragmentos(texto: str) -> list[str]:
    """Agrupa párrafos en fragmentos que no superen MAX_TOKENS_FRAGMENTO."""
    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    fragmentos, actual = [], ""
    for parrafo in parrafos:
        candidato = f"{actual}\n\n{parrafo}".strip()
        if actual and len(tokenizer.tokenize(candidato)) > MAX_TOKENS_FRAGMENTO:
            fragmentos.append(actual)
            actual = parrafo
        else:
            actual = candidato
    if actual:
        fragmentos.append(actual)
    return fragmentos


def resumir_texto(texto: str) -> str:
    fragmentos = _dividir_en_fragmentos(texto)
    if len(fragmentos) == 1:
        return _generar(texto)
    parciales = "\n\n".join(_generar(f, palabras=100) for f in fragmentos)
    return _generar(parciales)

@mcp.tool()
def resumir_archivo(ruta: str) -> str:
    """Resume en un párrafo, en español, el contenido de un archivo de texto.

    Args:
        ruta: ruta del archivo .txt a resumir. Si es relativa, se busca en la
            carpeta de este servidor (por ejemplo "ejemplo.txt").
    """
    if not os.path.isabs(ruta):
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta)
    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read()
    return resumir_texto(contenido)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
