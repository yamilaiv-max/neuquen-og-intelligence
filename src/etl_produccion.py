#!/usr/bin/env python3
"""ETL de producción de petróleo y gas de Neuquén.

Descarga las fuentes públicas SESCO, valida columnas, detecta el último
mes común cerrado, filtra Neuquén, conserva una ventana móvil de 12 meses
y genera archivos compactos para Airtable y Make.
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
    "fecha", "empresa", "areayacimiento", "idareayacimiento",
    "cuenca", "provincia", "concepto"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("etl_produccion")


def quitar_acentos(valor: object) -> str:
    texto = unicodedata.normalize("NFD", str(valor))
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def texto_normalizado(valor: object) -> str:
    return " ".join(quitar_acentos(valor).strip().lower().split())


def es_neuquen(valor: object) -> bool:
    return texto_normalizado(valor) == "neuquen"


def normalizar_recurso(valor: object) -> str:
    texto = texto_normalizado(valor).replace("_", " ")
    if "shale" in texto:
        return "Shale"
    if "tight" in texto:
        return "Tight"
    if "convencional" in texto:
        return "Convencional"
    raise ValueError(f"Concepto no reconocido: {valor!r}")


def descargar(url: str, destino: Path) -> None:
    LOG.info("Descargando %s", url)
    req = Request(url, headers={"User-Agent": "neuquen-og-intelligence/1.1"})
    with urlopen(req, timeout=300) as respuesta, destino.open("wb") as salida:
        while True:
            bloque = respuesta.read(1024 * 1024)
            if not bloque:
                break
            salida.write(bloque)
    LOG.info("Descarga finalizada: %.2f MB", destino.stat().st_size / 1_048_576)


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_fuente(ruta: Path, columna_valor: str, producto: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, low_memory=False)
    faltantes = sorted((COLUMNAS_BASE | {columna_valor}) - set(df.columns))
    if faltantes:
        raise ValueError(f"{producto}: faltan columnas {faltantes}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if df["fecha"].isna().any():
        raise ValueError(f"{producto}: existen fechas inválidas")

    df["Valor"] = pd.to_numeric(df[columna_valor], errors="coerce").fillna(0.0)
    df["Producto"] = producto
    df["Tipo Recurso"] = df["concepto"].map(normalizar_recurso)
    return df


def totales_mensuales_neuquen(df: pd.DataFrame) -> pd.Series:
    neuquen = df.loc[df["provincia"].map(es_neuquen)].copy()
    return neuquen.groupby("fecha")["Valor"].sum().sort_index()


def detectar_periodo_cerrado(
    petroleo: pd.DataFrame,
    gas: pd.DataFrame,
    umbral_completitud: float,
) -> tuple[pd.Timestamp, list[str], dict]:
    """Devuelve el último período común que supera el control de completitud.

    El mes candidato se compara con su mes anterior. Si petróleo o gas
    representa menos del umbral configurado, se considera una carga parcial
    y se retrocede un mes. El control puede retroceder más de una vez.
    """
    oil = totales_mensuales_neuquen(petroleo)
    gas_s = totales_mensuales_neuquen(gas)
    comunes = oil.index.intersection(gas_s.index).sort_values()
    if len(comunes) < 13:
        raise ValueError("No hay al menos 13 meses comunes para controlar completitud")

    candidato = comunes[-1]
    advertencias: list[str] = []
    controles: dict = {}

    while True:
        posicion = comunes.get_loc(candidato)
        if posicion == 0:
            raise ValueError("No existe mes anterior para validar completitud")
        anterior = comunes[posicion - 1]

        oil_actual = float(oil.loc[candidato])
        oil_anterior = float(oil.loc[anterior])
        gas_actual = float(gas_s.loc[candidato])
        gas_anterior = float(gas_s.loc[anterior])

        ratio_oil = oil_actual / oil_anterior if oil_anterior > 0 else 1.0
        ratio_gas = gas_actual / gas_anterior if gas_anterior > 0 else 1.0

        controles[candidato.strftime("%Y-%m")] = {
            "periodo_anterior": anterior.strftime("%Y-%m"),
            "petroleo_actual": round(oil_actual, 4),
            "petroleo_anterior": round(oil_anterior, 4),
            "ratio_petroleo": round(ratio_oil, 6),
            "gas_actual": round(gas_actual, 4),
            "gas_anterior": round(gas_anterior, 4),
            "ratio_gas": round(ratio_gas, 6),
            "umbral": umbral_completitud,
        }

        if ratio_oil >= umbral_completitud and ratio_gas >= umbral_completitud:
            break

        mensaje = (
            f"Período {candidato:%Y-%m} descartado por posible carga parcial. "
            f"Cobertura petróleo={ratio_oil:.1%}, gas={ratio_gas:.1%}, "
            f"umbral={umbral_completitud:.0%}."
        )
        advertencias.append(mensaje)
        LOG.warning(mensaje)
        candidato = anterior

        if comunes.get_loc(candidato) < 12:
            raise ValueError("No quedan 12 meses cerrados después del control")

    return candidato, advertencias, controles


def filtrar(df: pd.DataFrame, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.DataFrame:
    mascara = df["provincia"].map(es_neuquen) & df["fecha"].between(desde, hasta)
    return df.loc[mascara].copy()


def variacion(serie: pd.Series) -> float:
    anterior = float(serie.iloc[-2])
    actual = float(serie.iloc[-1])
    return np.nan if anterior == 0 else actual / anterior - 1


def operadora_principal(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    ranking = df.groupby("empresa")["Valor"].sum().sort_values(ascending=False)
    return "" if ranking.empty or ranking.iloc[0] <= 0 else str(ranking.index[0]).strip()


def construir_yacimientos(oil: pd.DataFrame, gas: pd.DataFrame, meses, act: str) -> pd.DataFrame:
    ids = sorted(
        set(oil["idareayacimiento"].dropna().astype(str))
        | set(gas["idareayacimiento"].dropna().astype(str))
    )
    filas = []
    for yid in ids:
        od = oil[oil["idareayacimiento"].astype(str) == yid]
        gd = gas[gas["idareayacimiento"].astype(str) == yid]
        ambos = pd.concat([od, gd], ignore_index=True)
        nombres = ambos["areayacimiento"].dropna().astype(str)
        cuencas = ambos["cuenca"].dropna().astype(str)
        nombre = nombres.mode().iloc[0] if not nombres.empty else yid
        cuenca = cuencas.mode().iloc[0] if not cuencas.empty else ""
        om = od.groupby("fecha")["Valor"].sum().reindex(meses, fill_value=0.0)
        gm = gd.groupby("fecha")["Valor"].sum().reindex(meses, fill_value=0.0)
        total_o, total_g = float(om.sum()), float(gm.sum())
        if total_o <= 0 and total_g <= 0:
            continue

        fila = {
            "ID Yacimiento": yid,
            "Yacimiento": nombre,
            "Cuenca": cuenca,
            "Provincia": "Neuquén",
            "Operadora Principal Petróleo": operadora_principal(od),
            "Operadora Principal Gas": operadora_principal(gd),
            "Período Desde": meses.min().strftime("%Y-%m-%d"),
            "Período Hasta": meses.max().strftime("%Y-%m-%d"),
            "Petróleo Total 12M": round(total_o, 4),
            "Gas Total 12M": round(total_g, 4),
            "Variación Petróleo Último Mes": variacion(om),
            "Variación Gas Último Mes": variacion(gm),
            "ID Actualización": act,
        }
        for i, mes in enumerate(meses):
            etiqueta = "M0" if i == 11 else f"M{i - 11}"
            fila[f"Período {etiqueta}"] = mes.strftime("%Y-%m-%d")
            fila[f"Petróleo {etiqueta}"] = round(float(om.loc[mes]), 4)
            fila[f"Gas {etiqueta}"] = round(float(gm.loc[mes]), 4)
        filas.append(fila)

    resultado = pd.DataFrame(filas).sort_values("ID Yacimiento").reset_index(drop=True)
    if resultado.empty or resultado["ID Yacimiento"].duplicated().any():
        raise ValueError("Salida de yacimientos vacía o con IDs duplicados")
    return resultado


def construir_series(fuentes: dict, meses, act: str) -> pd.DataFrame:
    filas = []
    unidades = {"Petróleo": "m3", "Gas": "miles de m3"}
    for producto, df in fuentes.items():
        tabla = (
            df.groupby(["fecha", "Tipo Recurso"])["Valor"].sum()
            .unstack(fill_value=0.0)
            .reindex(meses, fill_value=0.0)
            .reindex(columns=["Convencional", "Tight", "Shale"], fill_value=0.0)
        )
        for fecha, valores in tabla.iterrows():
            total = float(valores.sum())
            for recurso in ["Convencional", "Tight", "Shale"]:
                valor = float(valores[recurso])
                codigo_producto = "OIL" if producto == "Petróleo" else "GAS"
                filas.append({
                    "ID Serie": f"SER-{fecha:%Y-%m}-{codigo_producto}-{recurso.upper()}",
                    "ID Actualización": act,
                    "Período": fecha.strftime("%Y-%m-%d"),
                    "Tipo Período": "Mes histórico",
                    "Producto": producto,
                    "Tipo Recurso": recurso,
                    "Valor Producción": round(valor, 4),
                    "Unidad": unidades[producto],
                    "Participación": valor / total if total else np.nan,
                    "Es Proyección": False,
                    "Método Forecast": "",
                    "Límite Inferior": "",
                    "Límite Superior": "",
                    "Fecha Cálculo": "",
                })
    resultado = pd.DataFrame(filas)
    if len(resultado) != 72 or resultado["ID Serie"].duplicated().any():
        raise ValueError(f"Series inválidas: {len(resultado)} registros")
    return resultado


def construir_ranking(fuentes: dict, yac: pd.DataFrame, act: str, desde, hasta) -> pd.DataFrame:
    filas = []
    unidades = {"Petróleo": "m3", "Gas": "miles de m3"}
    columnas = {"Petróleo": "Petróleo Total 12M", "Gas": "Gas Total 12M"}
    col_op = {"Petróleo": "Operadora Principal Petróleo", "Gas": "Operadora Principal Gas"}

    for producto, df in fuentes.items():
        codigo = "PETROLEO" if producto == "Petróleo" else "GAS"
        ops = df.groupby("empresa")["Valor"].sum().sort_values(ascending=False).head(5)
        for pos, (empresa, valor) in enumerate(ops.items(), 1):
            filas.append({
                "ID Ranking": f"RANK-{hasta:%Y-%m}-OP-{codigo}-{pos:02d}",
                "ID Actualización": act,
                "Tipo Ranking": f"Top Operadoras {producto}",
                "Posición": pos, "Entidad": empresa, "ID Yacimiento": "",
                "Operadora": empresa, "Producción": round(float(valor), 4),
                "Cantidad Yacimientos": "", "Producto": producto,
                "Unidad": unidades[producto], "Período Desde": desde.strftime("%Y-%m-%d"),
                "Período Hasta": hasta.strftime("%Y-%m-%d"), "Fecha Cálculo": ""
            })

        top = yac.sort_values(columnas[producto], ascending=False).head(5)
        for pos, (_, r) in enumerate(top.iterrows(), 1):
            filas.append({
                "ID Ranking": f"RANK-{hasta:%Y-%m}-YAC-{codigo}-{pos:02d}",
                "ID Actualización": act,
                "Tipo Ranking": f"Top Yacimientos {producto}",
                "Posición": pos, "Entidad": r["Yacimiento"],
                "ID Yacimiento": r["ID Yacimiento"], "Operadora": r[col_op[producto]],
                "Producción": round(float(r[columnas[producto]]), 4),
                "Cantidad Yacimientos": "", "Producto": producto,
                "Unidad": unidades[producto], "Período Desde": desde.strftime("%Y-%m-%d"),
                "Período Hasta": hasta.strftime("%Y-%m-%d"), "Fecha Cálculo": ""
            })

    activos = pd.concat([
        df.loc[df["Valor"] > 0, ["empresa", "idareayacimiento"]]
        for df in fuentes.values()
    ]).drop_duplicates()
    cantidades = activos.groupby("empresa")["idareayacimiento"].nunique().sort_values(ascending=False).head(5)
    for pos, (empresa, cantidad) in enumerate(cantidades.items(), 1):
        filas.append({
            "ID Ranking": f"RANK-{hasta:%Y-%m}-OP-CANT-{pos:02d}",
            "ID Actualización": act,
            "Tipo Ranking": "Operadoras por Cantidad de Yacimientos",
            "Posición": pos, "Entidad": empresa, "ID Yacimiento": "",
            "Operadora": empresa, "Producción": "",
            "Cantidad Yacimientos": int(cantidad), "Producto": "Ambos",
            "Unidad": "yacimientos", "Período Desde": desde.strftime("%Y-%m-%d"),
            "Período Hasta": hasta.strftime("%Y-%m-%d"), "Fecha Cálculo": ""
        })

    resultado = pd.DataFrame(filas)
    if len(resultado) != 25 or resultado["ID Ranking"].duplicated().any():
        raise ValueError(f"Ranking inválido: {len(resultado)} registros")
    return resultado


def construir_indicadores(fuentes, yac, series, ranking, act, desde, hasta, advertencias) -> pd.DataFrame:
    participaciones = {}
    distribucion = {}
    evolucion = {}

    for producto in ["Petróleo", "Gas"]:
        datos = series[series["Producto"] == producto]
        por_tipo = datos.groupby("Tipo Recurso")["Valor Producción"].sum()
        total = float(por_tipo.sum())
        distribucion[producto] = {}
        for tipo in ["Convencional", "Tight", "Shale"]:
            valor = float(por_tipo.get(tipo, 0.0))
            participaciones[(producto, tipo)] = valor / total if total else None
            distribucion[producto][tipo] = {
                "valor": round(valor, 4),
                "participacion": participaciones[(producto, tipo)],
            }
        tabla = datos.pivot(index="Período", columns="Tipo Recurso", values="Valor Producción").fillna(0).sort_index()
        evolucion[producto] = {
            "periodos": tabla.index.tolist(),
            "convencional": tabla["Convencional"].round(4).tolist(),
            "tight": tabla["Tight"].round(4).tolist(),
            "shale": tabla["Shale"].round(4).tolist(),
            "no_convencional": (tabla["Tight"] + tabla["Shale"]).round(4).tolist(),
        }

    top_ops, top_yac = {}, {}
    for producto in ["Petróleo", "Gas"]:
        op = ranking[ranking["Tipo Ranking"] == f"Top Operadoras {producto}"].sort_values("Posición")
        ty = ranking[ranking["Tipo Ranking"] == f"Top Yacimientos {producto}"].sort_values("Posición")
        top_ops[producto] = [
            {"posicion": int(r["Posición"]), "empresa": r["Entidad"], "produccion": float(r["Producción"])}
            for _, r in op.iterrows()
        ]
        top_yac[producto] = [
            {"posicion": int(r["Posición"]), "id_yacimiento": r["ID Yacimiento"],
             "yacimiento": r["Entidad"], "operadora": r["Operadora"],
             "produccion": float(r["Producción"])}
            for _, r in ty.iterrows()
        ]

    productoras = set()
    for df in fuentes.values():
        productoras.update(df.loc[df["Valor"] > 0, "empresa"].dropna().astype(str))

    return pd.DataFrame([{
        "ID Actualización": act,
        "Período Desde": desde.strftime("%Y-%m-%d"),
        "Período Hasta": hasta.strftime("%Y-%m-%d"),
        "Provincia": "Neuquén",
        "Estado Indicadores": "Pendiente",
        "Total Petróleo 12M": round(float(fuentes["Petróleo"]["Valor"].sum()), 4),
        "Total Gas 12M": round(float(fuentes["Gas"]["Valor"].sum()), 4),
        "Cantidad Yacimientos": len(yac),
        "Cantidad Productoras": len(productoras),
        "Participación Convencional Oil": participaciones[("Petróleo", "Convencional")],
        "Participación Tight Oil": participaciones[("Petróleo", "Tight")],
        "Participación Shale Oil": participaciones[("Petróleo", "Shale")],
        "Participación Convencional Gas": participaciones[("Gas", "Convencional")],
        "Participación Tight Gas": participaciones[("Gas", "Tight")],
        "Participación Shale Gas": participaciones[("Gas", "Shale")],
        "Top Productoras JSON": json.dumps(top_ops, ensure_ascii=False),
        "Top 5 Yacimientos JSON": json.dumps(top_yac, ensure_ascii=False),
        "Evolución por Tipo de Recurso JSON": json.dumps(evolucion, ensure_ascii=False),
        "Distribución por Recurso JSON": json.dumps(distribucion, ensure_ascii=False),
        "Proyección 5 Semestres JSON": "",
        "Modelo de Predicción": "Pendiente de definir",
        "Resultado Auditoría Gemini": "Pendiente",
        "Resumen Auditoría Gemini": "",
        "Errores Críticos": 0,
        "Advertencias": len(advertencias),
        "Fecha Auditoría": "",
        "Indicadores Validados": False,
        "Decisión Humana Indicadores": "Pendiente",
        "Observaciones Validación": " | ".join(advertencias),
        "Responsable Validación": "",
        "Fecha Validación": "",
        "ID Ejecución": "",
        "Fecha Cálculo": "",
        "Mensaje Error": "",
    }])


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    LOG.info("Generado %s: %s registros", ruta.name, len(df))


def ejecutar(args) -> None:
    salida = Path(args.output_dir)
    salida.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="etl_neuquen_") as tmp:
        tmp = Path(tmp)
        ruta_oil = Path(args.petroleo_file) if args.petroleo_file else tmp / "petroleo.csv"
        ruta_gas = Path(args.gas_file) if args.gas_file else tmp / "gas.csv"
        if not args.petroleo_file:
            descargar(args.petroleo_url, ruta_oil)
        if not args.gas_file:
            descargar(args.gas_url, ruta_gas)

        oil_total = leer_fuente(ruta_oil, "cantidad_m3", "Petróleo")
        gas_total = leer_fuente(ruta_gas, "cantidad_mm3", "Gas")

        hasta, advertencias, controles = detectar_periodo_cerrado(
            oil_total, gas_total, args.umbral_completitud
        )
        desde = hasta - pd.DateOffset(months=11)
        meses = pd.date_range(desde, hasta, freq="MS")
        act = f"ACT-{hasta:%Y-%m}"

        oil = filtrar(oil_total, desde, hasta)
        gas = filtrar(gas_total, desde, hasta)
        fuentes = {"Petróleo": oil, "Gas": gas}

        yac = construir_yacimientos(oil, gas, meses, act)
        series = construir_series(fuentes, meses, act)
        ranking = construir_ranking(fuentes, yac, act, desde, hasta)
        indicadores = construir_indicadores(
            fuentes, yac, series, ranking, act, desde, hasta, advertencias
        )

        archivos = {
            "yacimientos": salida / "yacimientos_12m.csv",
            "ranking": salida / "ranking_productivo.csv",
            "series": salida / "series_productivas.csv",
            "indicadores": salida / "indicadores_produccion.csv",
        }
        guardar_csv(yac, archivos["yacimientos"])
        guardar_csv(ranking, archivos["ranking"])
        guardar_csv(series, archivos["series"])
        guardar_csv(indicadores, archivos["indicadores"])

        manifest = {
            "id_actualizacion": act,
            "periodo_desde": desde.strftime("%Y-%m-%d"),
            "periodo_hasta": hasta.strftime("%Y-%m-%d"),
            "provincia": "Neuquén",
            "estado": "procesado_con_advertencias" if advertencias else "procesado",
            "cantidad_yacimientos": len(yac),
            "cantidad_rankings": len(ranking),
            "cantidad_series": len(series),
            "cantidad_productoras": int(indicadores.iloc[0]["Cantidad Productoras"]),
            "advertencias": advertencias,
            "control_completitud": controles,
            "umbral_completitud": args.umbral_completitud,
            "fecha_procesamiento_utc": datetime.now(timezone.utc).isoformat(),
            "fuentes": {
                "petroleo_url": args.petroleo_url,
                "gas_url": args.gas_url,
                "petroleo_sha256": sha256(ruta_oil),
                "gas_sha256": sha256(ruta_gas),
            },
            "archivos": {
                clave: {"nombre": ruta.name, "sha256": sha256(ruta), "registros": len(df)}
                for (clave, ruta), df in zip(
                    archivos.items(), [yac, ranking, series, indicadores]
                )
            },
        }
        (salida / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOG.info("ETL OK: %s | %s a %s | %s yacimientos", act, desde.date(), hasta.date(), len(yac))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/output")
    p.add_argument("--petroleo-url", default=PETROLEO_URL)
    p.add_argument("--gas-url", default=GAS_URL)
    p.add_argument("--petroleo-file", default=os.getenv("PETROLEO_FILE"))
    p.add_argument("--gas-file", default=os.getenv("GAS_FILE"))
    p.add_argument("--umbral-completitud", type=float, default=0.80)
    return p


def main() -> int:
    try:
        ejecutar(parser().parse_args())
        return 0
    except Exception as exc:
        LOG.exception("ETL falló: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
