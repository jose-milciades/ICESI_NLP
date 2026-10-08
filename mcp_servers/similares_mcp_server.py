import logging
import math
import os
import re
from collections import Counter
from urllib.parse import urlparse

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer, util

# Modelo de embeddings multilingüe: se usa para extraer las frases clave del
# texto (estilo KeyBERT) y para ordenar los resultados por similitud semántica.
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
mcp = FastMCP("SimilaresMCP", host="localhost", port=9082)
# El cliente HTTP de ddgs registra cada petición a nivel INFO.
logging.getLogger("primp").setLevel(logging.WARNING)

STOPWORDS = set("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde
donde durante e el ella ellas ellos en entre era eran es esa esas ese eso esos
esta estaba estas este esto estos fue fueron ha han hasta hay la las le les lo
los mas más me mi muy no nos o otra otras otro otros para pero por porque que
qué se ser si sin sobre su sus también tan te tras tu un una uno unos y ya cada
todo todos toda todas desde hacia según sus son será sido siendo así bien
""".split())

DOMINIOS_BLOG = ("blogspot.", "wordpress.", "medium.com", "substack.com", "tumblr.")
DOMINIOS_VIDEO = ("youtube.com", "youtu.be", "vimeo.com")
EXTENSIONES_ARCHIVO = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")


def _leer(ruta: str) -> str:
    if not os.path.isabs(ruta):
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta)
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()


def _palabras(texto: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", texto)


def _es_contenido(palabra: str) -> bool:
    return len(palabra) >= 3 and palabra.lower() not in STOPWORDS


def _conceptos(texto: str) -> list[str]:
    """N-gramas de 1 y 2 palabras de contenido (sin stopwords ni números)."""
    candidatos = set()
    for oracion in re.split(r"[.!?;:\n()]+", texto):
        palabras = _palabras(oracion)
        for n in (1, 2):
            for i in range(len(palabras) - n + 1):
                grupo = palabras[i:i + n]
                if all(_es_contenido(p) for p in grupo):
                    candidatos.add(" ".join(grupo).lower())
    return list(candidatos)


def _entidades(texto: str) -> list[str]:
    """Nombres propios: secuencias de palabras en mayúscula, que pueden
    incluir conectores ("Tour de Francia", "Eddy Merckx"). Se descarta la
    primera palabra de cada oración salvo que forme parte de una secuencia."""
    entidades = set()
    patron = r"[A-ZÁÉÍÓÚÑ][\wáéíóúñü]+(?:\s+(?:de|del|la|y)?\s*[A-ZÁÉÍÓÚÑ][\wáéíóúñü]+)*"
    for oracion in re.split(r"(?<=[.!?:])\s+|\n+", texto):
        oracion = oracion.strip()
        for m in re.finditer(patron, oracion):
            palabras = m.group(0).split()
            # Quita conectores capitalizados al inicio ("Tras la Segunda...").
            while palabras and palabras[0].lower() in STOPWORDS | {"la", "de", "del", "y"}:
                palabras = palabras[1:]
            es_inicio = m.start() == 0 and len(palabras) == 1
            if palabras and not es_inicio:
                entidades.add(" ".join(palabras))
    return list(entidades)


def _mmr(candidatos: list[str], emb_doc, cantidad: int, diversidad: float) -> list[str]:
    """Maximal Marginal Relevance: elige frases relevantes para el documento
    penalizando las que son redundantes con las ya elegidas."""
    if not candidatos:
        return []
    emb_cand = embedder.encode(candidatos, convert_to_tensor=True)
    relevancia = util.cos_sim(emb_cand, emb_doc).squeeze(1)
    entre_si = util.cos_sim(emb_cand, emb_cand)
    elegidos = [int(relevancia.argmax())]
    while len(elegidos) < min(cantidad, len(candidatos)):
        puntajes = [
            float("-inf") if i in elegidos else
            (1 - diversidad) * float(relevancia[i])
            - diversidad * max(float(entre_si[i][j]) for j in elegidos)
            for i in range(len(candidatos))
        ]
        elegidos.append(max(range(len(candidatos)), key=puntajes.__getitem__))
    return [candidatos[i] for i in elegidos]


def _tema_principal(texto: str, emb_doc) -> str:
    """Palabra que mejor combina frecuencia en el texto y similitud con él."""
    frecuencia = Counter(p.lower() for p in _palabras(texto) if _es_contenido(p))
    palabras = [p for p, c in frecuencia.items() if c > 1] or list(frecuencia)
    emb = embedder.encode(palabras, convert_to_tensor=True)
    relevancia = util.cos_sim(emb, emb_doc).squeeze(1).tolist()
    puntajes = [r * math.log(1 + frecuencia[p]) for p, r in zip(palabras, relevancia)]
    return palabras[max(range(len(palabras)), key=puntajes.__getitem__)]


def extraer_consultas(texto: str) -> tuple[str, list[str], list[str]]:
    """Devuelve el tema principal, las frases clave y las consultas de búsqueda.
    Cada consulta se ancla al tema principal para no desviarse a subtemas."""
    emb_doc = embedder.encode(texto, convert_to_tensor=True)
    tema = _tema_principal(texto, emb_doc)
    conceptos = [c for c in _mmr(_conceptos(texto), emb_doc, 6, 0.6)
                 if tema not in c.split()][:4]
    entidades = _mmr(_entidades(texto), emb_doc, 4, 0.5)
    consultas = [f"{tema} {conceptos[0]}" if conceptos else tema]
    consultas += [f"{tema} {e}" for e in entidades[:3]]
    consultas += [f"{tema} {c}" for c in conceptos[1:3]]
    return tema, conceptos + entidades, consultas


def _clasificar(url: str, origen: str) -> str:
    dominio = urlparse(url).netloc.lower()
    ruta = urlparse(url).path.lower()
    if ruta.endswith(EXTENSIONES_ARCHIVO):
        return "archivo"
    if origen == "noticias":
        return "noticia"
    if "wikipedia.org" in dominio:
        return "wikipedia"
    if any(d in dominio for d in DOMINIOS_VIDEO):
        return "video"
    if any(d in dominio for d in DOMINIOS_BLOG) or "/blog" in ruta or dominio.startswith("blog."):
        return "blog"
    return "artículo/web"


def _buscar(consultas: list[str], por_consulta: int) -> list[dict]:
    resultados, titulos = {}, set()
    ddgs = DDGS()

    def agregar(items, origen, url_key="href"):
        for r in items or []:
            url = r.get(url_key) or r.get("url") or ""
            titulo = r.get("title", "").strip()
            # Se descartan anuncios, enlaces de redirección internos del
            # buscador y duplicados (por URL o por título).
            if (not url.startswith("http") or "bing.com/aclick" in url
                    or url in resultados or titulo.lower() in titulos):
                continue
            titulos.add(titulo.lower())
            resultados[url] = {
                "titulo": titulo,
                "url": url,
                "descripcion": (r.get("body") or "").strip(),
                "tipo": _clasificar(url, origen),
            }

    for consulta in consultas:
        try:
            agregar(ddgs.text(consulta, region="es-es", max_results=por_consulta), "web")
        except Exception:
            pass
    # Búsquedas específicas para documentos y noticias con la consulta principal.
    principal = consultas[0]
    for consulta, metodo, origen in (
        (f"{principal} filetype:pdf", ddgs.text, "web"),
        (f"{principal} blog", ddgs.text, "web"),
        (principal, ddgs.news, "noticias"),
    ):
        try:
            agregar(metodo(consulta, region="es-es", max_results=por_consulta), origen)
        except Exception:
            pass
    return list(resultados.values())


def _variar_tipos(resultados: list[dict], cantidad: int, minimo: float = 0.4) -> list[dict]:
    """Garantiza al menos un resultado relevante de cada tipo (archivo, blog,
    noticia...) y completa con los más similares. Mantiene el orden por similitud."""
    elegidos = []
    for tipo in dict.fromkeys(r["tipo"] for r in resultados):
        mejor = next(r for r in resultados if r["tipo"] == tipo)
        if mejor["similitud"] >= minimo:
            elegidos.append(mejor)
    for r in resultados:
        if len(elegidos) >= cantidad:
            break
        if r not in elegidos:
            elegidos.append(r)
    elegidos = elegidos[:cantidad]
    return sorted(elegidos, key=lambda r: r["similitud"], reverse=True)


def buscar_relacionados(texto: str, max_resultados: int = 10) -> str:
    tema, frases, consultas = extraer_consultas(texto)
    resultados = _buscar(consultas, por_consulta=8)
    if not resultados:
        return (f"Tema: {tema}. Frases clave: {', '.join(frases)}\n\n"
                "No se encontraron resultados en internet.")

    # Ordena por similitud semántica entre el texto y título + descripción.
    emb_doc = embedder.encode(texto, convert_to_tensor=True)
    emb_res = embedder.encode([f"{r['titulo']}. {r['descripcion']}" for r in resultados],
                              convert_to_tensor=True)
    for r, sim in zip(resultados, util.cos_sim(emb_res, emb_doc).squeeze(1).tolist()):
        r["similitud"] = sim
    resultados.sort(key=lambda r: r["similitud"], reverse=True)
    resultados = _variar_tipos(resultados, max_resultados)

    lineas = [f"Tema principal: {tema}",
              f"Frases clave detectadas: {', '.join(frases)}",
              f"Consultas realizadas: {' | '.join(consultas)}", "",
              f"Recursos relacionados ({min(max_resultados, len(resultados))}):"]
    for n, r in enumerate(resultados[:max_resultados], 1):
        lineas.append(f"{n}. [{r['tipo']}] {r['titulo']} (similitud {r['similitud']:.2f})")
        lineas.append(f"   {r['url']}")
        if r["descripcion"]:
            lineas.append(f"   {r['descripcion'][:200]}")
    return "\n".join(lineas)


@mcp.tool()
def buscar_similares(ruta: str, max_resultados: int = 10) -> str:
    """Busca en internet recursos relacionados con el contenido de un archivo
    de texto: páginas web, artículos, blogs, noticias y archivos (PDF, etc.).

    Extrae las frases clave del texto, consulta un buscador web y devuelve los
    resultados ordenados por similitud semántica con el texto original.

    Args:
        ruta: ruta del archivo .txt a analizar. Si es relativa, se busca en la
            carpeta de este servidor (por ejemplo "ejemplo.txt").
        max_resultados: cantidad máxima de recursos a devolver.
    """
    return buscar_relacionados(_leer(ruta), max_resultados)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
