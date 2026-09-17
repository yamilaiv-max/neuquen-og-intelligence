#!/usr/bin/env python3
"""ETL de producción de petróleo y gas de Neuquén.

Descarga las fuentes públicas SESCO, valida su estructura, identifica el
último período común, filtra Neuquén y genera archivos compactos para
Airtable y Make.

Salidas:
- data/output/yacimientos_12m.csv
- data/output/ranking_productivo.csv
- data/output/series_productivas.csv
- data/output/indicadores_produccion.csv
- data/output/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


PETROLEO_URL = (
    "http://datos.energia.gob.ar/dataset/590d1284-fd6d-4686-afd8-b3da5d90a6e9/"
    "resource/83a2b597-b087-4815-b17d-cd70990d6a79/download/"
    "produccin-petrleo-sesco-tight-y-shale-captulo-iv-por-yacimiento.csv"
)

GAS_URL = (
    "http://datos.energia.gob.ar/dataset/590d1284-fd6d-4686-afd8-b3da5d90a6e9/"
    "resource/931cfb07-37b7-414a-ae8b-528dff6f9f14/download/"
    "produccin-gas-sesco-tight-y-shale-captulo-iv-por-yacimiento.csv"
)

COLUMNAS_BASE = {
    "anio",
    "mes",
    "indice_tiempo",
    "fecha",
    "empresa",
    "areapermisoconcesion",
    "idareapermisoconcesion",
    "areayacimiento",
    "idareayacimiento",
    "cuenca",
    "provincia",
    "concepto",
}

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("etl_produccion")


def remover_acentos(texto: object) -> str:
    normalizado = unicodedata.normalize("NFD", str(texto))
    return "".join(c for c in normalizado if unicodedata.category(c) != "Mn")


def normalizar_texto(texto: object) -> str:
    return " ".join(remover_acentos(texto).strip().lower().split())


def normalizar_recurso(concepto: object) -> str:
    valor = normalizar_texto(concepto).replace("_", " ")
    if "shale" in valor:
        return "Shale"
    if "tight" in valor:
        return "Tight"
    if "convencional" in valor:
        return "Convencional"
    raise ValueError(f"Concepto productivo no reconocido: {concepto!r}")


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def descargar_archivo(url: str, destino: Path) -> None:
    LOGGER.info("Descargando %s", url)
    solicitud = Request(url, headers={"User-Agent": "neuquen-og-intelligence/1.0"})
    with urlopen(solicitud, timeout=300) as respuesta, destino.open("wb") as salida:
        while True:
            bloque = respuesta.read(1024 * 1024)
            if not bloque:
                break
            salida.write(bloque)
    LOGGER.info("Archivo descargado: %s (%.2f MB)", destino, destino.stat().st_size / 1_048_576)


def validar_columnas(df: pd.DataFrame, columna_valor: str, nombre: str) -> None:
    requeridas = COLUMNAS_BASE | {columna_valor}
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f"{nombre}: faltan columnas obligatorias: {faltantes}")


def leer_fuente(ruta: Path, columna_valor: str, producto: str) -> pd.DataFrame:
    LOGGER.info("Leyendo fuente %s", producto)
    df = pd.read_csv(ruta, low_memory=False)
    validar_columnas(df, columna_valor, producto)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["Valor"] = pd.to_numeric(df[columna_valor], errors="coerce").fillna(0.0)
    df["Producto"] = producto
    df["Tipo Recurso"] = df["concepto"].map(normalizar_recurso)

    invalidas = int(df["fecha"].isna().sum())
    if invalidas:
        raise ValueError(f"{producto}: se encontraron {invalidas} fechas inválidas")

    return df


def es_neuquen(valor: object) -> bool:
    return normalizar_texto(valor) == "neuquen"


def obtener_ventana_comun(
    petroleo: pd.DataFrame, gas: pd.DataFrame, meses: int = 12
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    max_petroleo = petroleo["fecha"].max()
    max_gas = gas["fecha"].max()
    periodo_hasta = min(max_petroleo, max_gas)
    periodo_desde = periodo_hasta - pd.DateOffset(months=meses - 1)

    if max_petroleo != max_gas:
        LOGGER.warning(
            "Las fuentes tienen períodos máximos diferentes. Petróleo=%s, Gas=%s. "
            "Se utilizará el último período común=%s.",
            max_petroleo.date(),
            max_gas.date(),
            periodo_hasta.date(),
        )

    return periodo_desde, periodo_hasta


def filtrar_ventana(
    df: pd.DataFrame, periodo_desde: pd.Timestamp, periodo_hasta: pd.Timestamp
) -> pd.DataFrame:
    provincia = df["provincia"].map(es_neuquen)
    ventana = df["fecha"].between(periodo_desde, periodo_hasta)
    salida = df.loc[provincia & ventana].copy()
    LOGGER.info(
        "%s: %s filas después de filtrar Neuquén y la ventana temporal",
        salida["Producto"].iloc[0] if not salida.empty else "Fuente",
        len(salida),
    )
    return salida


def variacion_ultimo_mes(serie: pd.Series) -> float:
    anterior = float(serie.iloc[-2])
    actual = float(serie.iloc[-1])
    if anterior == 0:
        return np.nan
    return actual / anterior - 1


def operadora_principal(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    agrupado = df.groupby("empresa", dropna=True)["Valor"].sum().sort_values(ascending=False)
    if agrupado.empty or float(agrupado.iloc[0]) <= 0:
        return ""
    return str(agrupado.index[0]).strip()


def construir_yacimientos(
    petroleo: pd.DataFrame,
    gas: pd.DataFrame,
    meses: pd.DatetimeIndex,
    id_actualizacion: str,
) -> pd.DataFrame:
    ids = sorted(
        set(petroleo["idareayacimiento"].dropna().astype(str))
        | set(gas["idareayacimiento"].dropna().astype(str))
    )
    registros = []

    for id_yacimiento in ids:
        oil = petroleo[petroleo["idareayacimiento"].astype(str) == id_yacimiento]
        gas_y = gas[gas["idareayacimiento"].astype(str) == id_yacimiento]
        ambos = pd.concat([oil, gas_y], ignore_index=True)

        nombres = ambos["areayacimiento"].dropna().astype(str)
        nombre = nombres.mode().iloc[0] if not nombres.empty else id_yacimiento
        cuencas = ambos["cuenca"].dropna().astype(str)
        cuenca = cuencas.mode().iloc[0] if not cuencas.empty else ""

        oil_mensual = oil.groupby("fecha")["Valor"].sum().reindex(meses, fill_value=0.0)
        gas_mensual = gas_y.groupby("fecha")["Valor"].sum().reindex(meses, fill_value=0.0)

        total_oil = float(oil_mensual.sum())
        total_gas = float(gas_mensual.sum())

        # Solo se conservan yacimientos activos durante la ventana móvil.
        if total_oil <= 0 and total_gas <= 0:
            continue

        registro = {
            "ID Yacimiento": id_yacimiento,
            "Yacimiento": nombre,
            "Cuenca": cuenca,
            "Provincia": "Neuquén",
            "Operadora Principal Petróleo": operadora_principal(oil),
            "Operadora Principal Gas": operadora_principal(gas_y),
            "Período Desde": meses.min().strftime("%Y-%m-%d"),
            "Período Hasta": meses.max().strftime("%Y-%m-%d"),
            "Petróleo Total 12M": round(total_oil, 4),
            "Gas Total 12M": round(total_gas, 4),
            "Variación Petróleo Último Mes": variacion_ultimo_mes(oil_mensual),
            "Variación Gas Último Mes": variacion_ultimo_mes(gas_mensual),
            "ID Actualización": id_actualizacion,
        }

        for indice, mes in enumerate(meses):
            etiqueta = "M0" if indice == 11 else f"M{indice - 11}"
            registro[f"Período {etiqueta}"] = mes.strftime("%Y-%m-%d")
            registro[f"Petróleo {etiqueta}"] = round(float(oil_mensual.loc[mes]), 4)
            registro[f"Gas {etiqueta}"] = round(float(gas_mensual.loc[mes]), 4)

        registros.append(registro)

    resultado = pd.DataFrame(registros).sort_values("ID Yacimiento").reset_index(drop=True)
    if resultado["ID Yacimiento"].duplicated().any():
        raise ValueError("Se generaron ID Yacimiento duplicados")
    return resultado


def construir_series(
    fuentes: Dict[str, pd.DataFrame],
    meses: pd.DatetimeIndex,
    id_actualizacion: str,
) -> pd.DataFrame:
    registros = []
    unidades = {"Petróleo": "m3", "Gas": "miles de m3"}

    for producto, df in fuentes.items():
        tabla = (
            df.groupby(["fecha", "Tipo Recurso"])["Valor"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(meses, fill_value=0.0)
            .reindex(columns=["Convencional", "Tight", "Shale"], fill_value=0.0)
        )

        for periodo, fila in tabla.iterrows():
            total = float(fila.sum())
            for recurso in ["Convencional", "Tight", "Shale"]:
                valor = float(fila[recurso])
                registros.append(
                    {
                        "ID Serie": (
                            f"SER-{periodo:%Y-%m}-{producto.upper()}-{recurso.upper()}"
                            .replace("Ó", "O")
                            .replace("É", "E")
                            .replace(" ", "_")
                        ),
                        "ID Actualización": id_actualizacion,
                        "Período": periodo.strftime("%Y-%m-%d"),
                        "Tipo Período": "Mes histórico",
                        "Producto": producto,
                        "Tipo Recurso": recurso,
                        "Valor Producción": round(valor, 4),
                        "Unidad": unidades[producto],
                        "Participación": (valor / total) if total else np.nan,
                        "Es Proyección": False,
                        "Método Forecast": "",
                        "Límite Inferior": "",
                        "Límite Superior": "",
                        "Fecha Cálculo": "",
                    }
                )

    resultado = pd.DataFrame(registros)
    if resultado["ID Serie"].duplicated().any():
        raise ValueError("Se generaron ID Serie duplicados")
    return resultado


def construir_ranking(
    fuentes: Dict[str, pd.DataFrame],
    yacimientos: pd.DataFrame,
    id_actualizacion: str,
    periodo_desde: pd.Timestamp,
    periodo_hasta: pd.Timestamp,
) -> pd.DataFrame:
    registros = []
    unidades = {"Petróleo": "m3", "Gas": "miles de m3"}
    columnas_totales = {"Petróleo": "Petróleo Total 12M", "Gas": "Gas Total 12M"}
    columnas_operadora = {
        "Petróleo": "Operadora Principal Petróleo",
        "Gas": "Operadora Principal Gas",
    }

    for producto, df in fuentes.items():
        top_operadoras = df.groupby("empresa")["Valor"].sum().sort_values(ascending=False).head(5)
        for posicion, (empresa, produccion) in enumerate(top_operadoras.items(), start=1):
            registros.append(
                {
                    "ID Ranking": f"RANK-{periodo_hasta:%Y-%m}-OP-{producto.upper()}-{posicion:02d}".replace("Ó", "O"),
                    "ID Actualización": id_actualizacion,
                    "Tipo Ranking": f"Top Operadoras {producto}",
                    "Posición": posicion,
                    "Entidad": empresa,
                    "ID Yacimiento": "",
                    "Operadora": empresa,
                    "Producción": round(float(produccion), 4),
                    "Cantidad Yacimientos": "",
                    "Producto": producto,
                    "Unidad": unidades[producto],
                    "Período Desde": periodo_desde.strftime("%Y-%m-%d"),
                    "Período Hasta": periodo_hasta.strftime("%Y-%m-%d"),
                    "Fecha Cálculo": "",
                }
            )

        top_yacimientos = yacimientos.sort_values(columnas_totales[producto], ascending=False).head(5)
        for posicion, (_, fila) in enumerate(top_yacimientos.iterrows(), start=1):
            registros.append(
                {
                    "ID Ranking": f"RANK-{periodo_hasta:%Y-%m}-YAC-{producto.upper()}-{posicion:02d}".replace("Ó", "O"),
                    "ID Actualización": id_actualizacion,
                    "Tipo Ranking": f"Top Yacimientos {producto}",
                    "Posición": posicion,
                    "Entidad": fila["Yacimiento"],
                    "ID Yacimiento": fila["ID Yacimiento"],
                    "Operadora": fila[columnas_operadora[producto]],
                    "Producción": round(float(fila[columnas_totales[producto]]), 4),
                    "Cantidad Yacimientos": "",
                    "Producto": producto,
                    "Unidad": unidades[producto],
                    "Período Desde": periodo_desde.strftime("%Y-%m-%d"),
                    "Período Hasta": periodo_hasta.strftime("%Y-%m-%d"),
                    "Fecha Cálculo": "",
                }
            )

    activos = []
    for df in fuentes.values():
        activos.append(df.loc[df["Valor"] > 0, ["empresa", "idareayacimiento"]])
    activos_df = pd.concat(activos).drop_duplicates()
    cantidades = (
        activos_df.groupby("empresa")["idareayacimiento"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
    )
    for posicion, (empresa, cantidad) in enumerate(cantidades.items(), start=1):
        registros.append(
            {
                "ID Ranking": f"RANK-{periodo_hasta:%Y-%m}-OP-CANT-{posicion:02d}",
                "ID Actualización": id_actualizacion,
                "Tipo Ranking": "Operadoras por Cantidad de Yacimientos",
                "Posición": posicion,
                "Entidad": empresa,
                "ID Yacimiento": "",
                "Operadora": empresa,
                "Producción": "",
                "Cantidad Yacimientos": int(cantidad),
                "Producto": "Ambos",
                "Unidad": "yacimientos",
                "Período Desde": periodo_desde.strftime("%Y-%m-%d"),
                "Período Hasta": periodo_hasta.strftime("%Y-%m-%d"),
                "Fecha Cálculo": "",
            }
        )

    resultado = pd.DataFrame(registros)
    if resultado["ID Ranking"].duplicated().any():
        raise ValueError("Se generaron ID Ranking duplicados")
    return resultado


def json_rankings(ranking: pd.DataFrame, tipo_base: str) -> str:
    salida = {}
    for producto in ["Petróleo", "Gas"]:
        datos = ranking[ranking["Tipo Ranking"] == f"{tipo_base} {producto}"]
        elementos = []
        for _, fila in datos.sort_values("Posición").iterrows():
            elemento = {
                "posicion": int(fila["Posición"]),
                "produccion": float(fila["Producción"]),
            }
            if tipo_base == "Top Operadoras":
                elemento["empresa"] = fila["Entidad"]
            else:
                elemento.update(
                    {
                        "id_yacimiento": fila["ID Yacimiento"],
                        "yacimiento": fila["Entidad"],
                        "operadora": fila["Operadora"],
                    }
                )
            elementos.append(elemento)
        salida[producto] = elementos
    return json.dumps(salida, ensure_ascii=False)


def construir_indicadores(
    fuentes: Dict[str, pd.DataFrame],
    yacimientos: pd.DataFrame,
    series: pd.DataFrame,
    ranking: pd.DataFrame,
    id_actualizacion: str,
    periodo_desde: pd.Timestamp,
    periodo_hasta: pd.Timestamp,
) -> pd.DataFrame:
    totales = {
        producto: float(df["Valor"].sum()) for producto, df in fuentes.items()
    }
    productoras = set()
    for df in fuentes.values():
        productoras.update(df.loc[df["Valor"] > 0, "empresa"].dropna().astype(str))

    participaciones = {}
    distribucion = {}
    evolucion = {}

    for producto in ["Petróleo", "Gas"]:
        datos = series[series["Producto"] == producto].copy()
        por_recurso = datos.groupby("Tipo Recurso")["Valor Producción"].sum()
        total = float(por_recurso.sum())
        distribucion[producto] = {}

        for recurso in ["Convencional", "Tight", "Shale"]:
            valor = float(por_recurso.get(recurso, 0.0))
            participacion = valor / total if total else None
            participaciones[(producto, recurso)] = participacion
            distribucion[producto][recurso] = {
                "valor": round(valor, 4),
                "participacion": participacion,
            }

        tabla = datos.pivot(index="Período", columns="Tipo Recurso", values="Valor Producción").fillna(0)
        tabla = tabla.sort_index()
        evolucion[producto] = {
            "periodos": tabla.index.tolist(),
            "convencional": tabla.get("Convencional", pd.Series(dtype=float)).round(4).tolist(),
            "tight": tabla.get("Tight", pd.Series(dtype=float)).round(4).tolist(),
            "shale": tabla.get("Shale", pd.Series(dtype=float)).round(4).tolist(),
            "no_convencional": (
                tabla.get("Tight", 0) + tabla.get("Shale", 0)
            ).round(4).tolist(),
        }

    registro = {
        "ID Actualización": id_actualizacion,
        "Período Desde": periodo_desde.strftime("%Y-%m-%d"),
        "Período Hasta": periodo_hasta.strftime("%Y-%m-%d"),
        "Provincia": "Neuquén",
        "Estado Indicadores": "Pendiente",
        "Total Petróleo 12M": round(totales["Petróleo"], 4),
        "Total Gas 12M": round(totales["Gas"], 4),
        "Cantidad Yacimientos": int(len(yacimientos)),
        "Cantidad Productoras": int(len(productoras)),
        "Participación Convencional Oil": participaciones[("Petróleo", "Convencional")],
        "Participación Tight Oil": participaciones[("Petróleo", "Tight")],
        "Participación Shale Oil": participaciones[("Petróleo", "Shale")],
        "Participación Convencional Gas": participaciones[("Gas", "Convencional")],
        "Participación Tight Gas": participaciones[("Gas", "Tight")],
        "Participación Shale Gas": participaciones[("Gas", "Shale")],
        "Top Productoras JSON": json_rankings(ranking, "Top Operadoras"),
        "Top 5 Yacimientos JSON": json_rankings(ranking, "Top Yacimientos"),
        "Evolución por Tipo de Recurso JSON": json.dumps(evolucion, ensure_ascii=False),
        "Distribución por Recurso JSON": json.dumps(distribucion, ensure_ascii=False),
        "Proyección 5 Semestres JSON": "",
        "Modelo de Predicción": "Pendiente de definir",
        "Resultado Auditoría Gemini": "Pendiente",
        "Resumen Auditoría Gemini": "",
        "Errores Críticos": 0,
        "Advertencias": 0,
        "Fecha Auditoría": "",
        "Indicadores Validados": False,
        "Decisión Humana Indicadores": "Pendiente",
        "Observaciones Validación": "",
        "Responsable Validación": "",
        "Fecha Validación": "",
        "ID Ejecución": "",
        "Fecha Cálculo": "",
        "Mensaje Error": "",
    }
    return pd.DataFrame([registro])


def escribir_csv(df: pd.DataFrame, ruta: Path) -> None:
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    LOGGER.info("Generado %s: %s registros", ruta.name, len(df))


def validar_salidas(
    yacimientos: pd.DataFrame,
    ranking: pd.DataFrame,
    series: pd.DataFrame,
    indicadores: pd.DataFrame,
) -> None:
    if yacimientos.empty:
        raise ValueError("La salida de yacimientos está vacía")
    if len(ranking) != 25:
        raise ValueError(f"Se esperaban 25 registros de ranking y se generaron {len(ranking)}")
    if len(series) != 72:
        raise ValueError(f"Se esperaban 72 registros de series y se generaron {len(series)}")
    if len(indicadores) != 1:
        raise ValueError("La salida de indicadores debe contener exactamente una fila")


def ejecutar(args: argparse.Namespace) -> None:
    salida = Path(args.output_dir)
    salida.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="etl_neuquen_") as temporal:
        temporal = Path(temporal)
        ruta_petroleo = Path(args.petroleo_file) if args.petroleo_file else temporal / "petroleo.csv"
        ruta_gas = Path(args.gas_file) if args.gas_file else temporal / "gas.csv"

        if not args.petroleo_file:
            descargar_archivo(args.petroleo_url, ruta_petroleo)
        if not args.gas_file:
            descargar_archivo(args.gas_url, ruta_gas)

        petroleo_completo = leer_fuente(ruta_petroleo, "cantidad_m3", "Petróleo")
        gas_completo = leer_fuente(ruta_gas, "cantidad_mm3", "Gas")

        periodo_desde, periodo_hasta = obtener_ventana_comun(
            petroleo_completo, gas_completo, meses=12
        )
        meses = pd.date_range(periodo_desde, periodo_hasta, freq="MS")
        if len(meses) != 12:
            raise ValueError(f"La ventana calculada no contiene 12 meses: {len(meses)}")

        petroleo = filtrar_ventana(petroleo_completo, periodo_desde, periodo_hasta)
        gas = filtrar_ventana(gas_completo, periodo_desde, periodo_hasta)
        fuentes = {"Petróleo": petroleo, "Gas": gas}

        id_actualizacion = f"ACT-{periodo_hasta:%Y-%m}"

        yacimientos = construir_yacimientos(petroleo, gas, meses, id_actualizacion)
        series = construir_series(fuentes, meses, id_actualizacion)
        ranking = construir_ranking(
            fuentes, yacimientos, id_actualizacion, periodo_desde, periodo_hasta
        )
        indicadores = construir_indicadores(
            fuentes,
            yacimientos,
            series,
            ranking,
            id_actualizacion,
            periodo_desde,
            periodo_hasta,
        )

        validar_salidas(yacimientos, ranking, series, indicadores)

        archivos = {
            "yacimientos": salida / "yacimientos_12m.csv",
            "ranking": salida / "ranking_productivo.csv",
            "series": salida / "series_productivas.csv",
            "indicadores": salida / "indicadores_produccion.csv",
        }
        escribir_csv(yacimientos, archivos["yacimientos"])
        escribir_csv(ranking, archivos["ranking"])
        escribir_csv(series, archivos["series"])
        escribir_csv(indicadores, archivos["indicadores"])

        manifest = {
            "id_actualizacion": id_actualizacion,
            "periodo_desde": periodo_desde.strftime("%Y-%m-%d"),
            "periodo_hasta": periodo_hasta.strftime("%Y-%m-%d"),
            "provincia": "Neuquén",
            "estado": "procesado",
            "cantidad_yacimientos": int(len(yacimientos)),
            "cantidad_rankings": int(len(ranking)),
            "cantidad_series": int(len(series)),
            "cantidad_productoras": int(indicadores.iloc[0]["Cantidad Productoras"]),
            "fecha_procesamiento_utc": datetime.now(timezone.utc).isoformat(),
            "fuentes": {
                "petroleo_url": args.petroleo_url,
                "gas_url": args.gas_url,
                "petroleo_sha256": sha256_archivo(ruta_petroleo),
                "gas_sha256": sha256_archivo(ruta_gas),
            },
            "archivos": {
                clave: {
                    "nombre": ruta.name,
                    "sha256": sha256_archivo(ruta),
                    "registros": {
                        "yacimientos": len(yacimientos),
                        "ranking": len(ranking),
                        "series": len(series),
                        "indicadores": len(indicadores),
                    }[clave],
                }
                for clave, ruta in archivos.items()
            },
        }

        ruta_manifest = salida / "manifest.json"
        ruta_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Generado manifest.json")

        LOGGER.info(
            "ETL finalizado: %s | período %s a %s | %s yacimientos",
            id_actualizacion,
            periodo_desde.date(),
            periodo_hasta.date(),
            len(yacimientos),
        )


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ETL de producción de Neuquén")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--petroleo-url", default=PETROLEO_URL)
    parser.add_argument("--gas-url", default=GAS_URL)
    parser.add_argument(
        "--petroleo-file",
        default=os.getenv("PETROLEO_FILE"),
        help="Archivo local opcional para pruebas",
    )
    parser.add_argument(
        "--gas-file",
        default=os.getenv("GAS_FILE"),
        help="Archivo local opcional para pruebas",
    )
    return parser


def main() -> int:
    try:
        args = crear_parser().parse_args()
        ejecutar(args)
        return 0
    except Exception as error:
        LOGGER.exception("El proceso ETL terminó con error: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
