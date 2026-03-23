import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from llama_cpp import Llama


TIER_PRESETS = {
    "bronze": {
        "n_ctx": 2048,
        "max_tokens": 220,
        "temperature": 0.2,
        "top_p": 0.85,
        "repeat_penalty": 1.08,
        "n_threads": 6,
    },
    "silver": {
        "n_ctx": 3072,
        "max_tokens": 320,
        "temperature": 0.3,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "n_threads": 6,
    },
    "gold": {
        "n_ctx": 4096,
        "max_tokens": 450,
        "temperature": 0.35,
        "top_p": 0.92,
        "repeat_penalty": 1.12,
        "n_threads": 6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plant descriptors with local Phi-3.")
    parser.add_argument("--tier", required=True, choices=sorted(TIER_PRESETS.keys()))
    parser.add_argument("--sample-size", type=int, default=0, help="0 means process all eligible plants.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-path", default=r"C:\models\Phi-3-mini-4k-instruct-q4.gguf")
    parser.add_argument("--plant-csv", default="eia860_plant.csv")
    parser.add_argument("--generator-csv", default="eia860_generator.csv")
    parser.add_argument("--emissions-csv", default="eia860_emissions_control_equipment.csv")
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("plant_descriptor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_tables(plant_path: Path, generator_path: Path, emissions_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plant_cols = [
        "eiap_plant_id",
        "eiap_plant_name",
        "eiap_utility_name",
        "eiap_city",
        "eiap_state",
        "eiap_sector_desc",
        "eiap_regulatory_status_desc",
        "eiap_nerc_region_desc",
    ]
    generator_cols = [
        "eiag_plant_id",
        "eiag_generator_id",
        "eiag_operating_date",
        "eiag_energy_source_1_fuel_category",
        "eiag_energy_source_1_desc",
        "eiag_technology_desc",
        "eiag_status_desc",
    ]
    emissions_cols = [
        "eiaece_plant_id",
        "eiaece_equipment_id",
        "eiaece_inservice_date",
        "eiaece_equipment_type_desc",
        "eiaece_status_desc",
    ]

    plant_df = pd.read_csv(plant_path, usecols=plant_cols, low_memory=False)
    generator_df = pd.read_csv(generator_path, usecols=generator_cols, low_memory=False)
    emissions_df = pd.read_csv(emissions_path, usecols=emissions_cols, low_memory=False)

    generator_df["eiag_operating_date"] = pd.to_datetime(generator_df["eiag_operating_date"], errors="coerce")
    emissions_df["eiaece_inservice_date"] = pd.to_datetime(emissions_df["eiaece_inservice_date"], errors="coerce")
    return plant_df, generator_df, emissions_df


def eligible_plant_ids(generator_df: pd.DataFrame, emissions_df: pd.DataFrame) -> List[int]:
    ng_plants = set(
        generator_df.loc[
            generator_df["eiag_energy_source_1_fuel_category"].eq("Natural Gas"),
            "eiag_plant_id",
        ].dropna().astype(int)
    )
    scr_plants = set(
        emissions_df.loc[
            emissions_df["eiaece_equipment_type_desc"].eq("Selective catalytic reduction"),
            "eiaece_plant_id",
        ].dropna().astype(int)
    )
    return sorted(ng_plants.intersection(scr_plants))


def summarize_age(series: pd.Series, today_year: int) -> Dict[str, float]:
    years = today_year - series.dt.year
    years = years.dropna()
    if years.empty:
        return {"min": float("nan"), "max": float("nan"), "avg": float("nan")}
    return {"min": float(years.min()), "max": float(years.max()), "avg": float(years.mean())}


def build_plant_bullets(
    plant_row: pd.Series,
    generators: pd.DataFrame,
    emissions: pd.DataFrame,
    today_year: int,
) -> List[str]:
    bullets: List[str] = []
    plant_name = plant_row["eiap_plant_name"]
    location = f"{plant_row.get('eiap_city', '')}, {plant_row.get('eiap_state', '')}".strip(", ")
    bullets.append(f"- Plant: {plant_name} ({int(plant_row['eiap_plant_id'])}) in {location}.")
    bullets.append(f"- Utility owner: {plant_row.get('eiap_utility_name', 'Unknown')}.")

    gen_age = summarize_age(generators["eiag_operating_date"], today_year)
    gen_count = len(generators)
    ng_count = int(generators["eiag_energy_source_1_fuel_category"].eq("Natural Gas").sum())
    bullets.append(
        f"- Generators: {gen_count} total, including {ng_count} natural-gas units. "
        f"Operating age range {int(gen_age['min']) if pd.notna(gen_age['min']) else 'NA'}-"
        f"{int(gen_age['max']) if pd.notna(gen_age['max']) else 'NA'} years, "
        f"average {gen_age['avg']:.1f} years."
        if pd.notna(gen_age["avg"])
        else f"- Generators: {gen_count} total, including {ng_count} natural-gas units. Age unavailable."
    )

    grouped_gen = (
        generators.groupby(["eiag_energy_source_1_desc", "eiag_technology_desc"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    for _, row in grouped_gen.head(6).iterrows():
        fuel = row["eiag_energy_source_1_desc"] if pd.notna(row["eiag_energy_source_1_desc"]) else "Unknown fuel"
        tech = row["eiag_technology_desc"] if pd.notna(row["eiag_technology_desc"]) else "Unknown technology"
        sub = generators[
            generators["eiag_energy_source_1_desc"].fillna("Unknown fuel").eq(fuel)
            & generators["eiag_technology_desc"].fillna("Unknown technology").eq(tech)
        ]
        age = summarize_age(sub["eiag_operating_date"], today_year)
        if pd.notna(age["avg"]):
            bullets.append(f"- Generator mix: {int(row['count'])} x {fuel} / {tech}, avg age {age['avg']:.1f} years.")
        else:
            bullets.append(f"- Generator mix: {int(row['count'])} x {fuel} / {tech}.")

    if not emissions.empty:
        emi_age = summarize_age(emissions["eiaece_inservice_date"], today_year)
        bullets.append(
            f"- Emissions controls: {len(emissions)} devices. "
            f"In-service age range {int(emi_age['min']) if pd.notna(emi_age['min']) else 'NA'}-"
            f"{int(emi_age['max']) if pd.notna(emi_age['max']) else 'NA'} years, "
            f"average {emi_age['avg']:.1f} years."
            if pd.notna(emi_age["avg"])
            else f"- Emissions controls: {len(emissions)} devices. Age unavailable."
        )
        type_counts = emissions["eiaece_equipment_type_desc"].fillna("Unknown type").value_counts().head(6)
        for eq_type, count in type_counts.items():
            sub = emissions[emissions["eiaece_equipment_type_desc"].fillna("Unknown type").eq(eq_type)]
            age = summarize_age(sub["eiaece_inservice_date"], today_year)
            if pd.notna(age["avg"]):
                bullets.append(f"- Emissions mix: {count} x {eq_type}, avg age {age['avg']:.1f} years.")
            else:
                bullets.append(f"- Emissions mix: {count} x {eq_type}.")

        status_counts = emissions["eiaece_status_desc"].fillna("Unknown status").value_counts()
        status_text = "; ".join(f"{k}: {v}" for k, v in status_counts.items())
        bullets.append(f"- Emissions control status: {status_text}.")

    return bullets


def build_prompt(bullets: List[str]) -> str:
    return "\n".join(
        [
            "You are supporting a technical sales team at an industrial services company.",
            "Write a concise, factual plant profile from the bullets below.",
            "Focus on generator count/type/age and emissions-control count/type/age.",
            "Highlight likely maintenance, retrofit, turnaround, catalyst, and balance-of-plant service opportunities.",
            "Use 2 short paragraphs plus a final bullet list of 3-5 actionable sales angles.",
            "",
            "Plant context bullets:",
            *bullets,
        ]
    )


def run(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"plant_descriptors_{args.tier}_{timestamp}.csv"
    log_path = output_dir / f"plant_descriptors_{args.tier}_{timestamp}.log"
    logger = setup_logging(log_path)

    tier_params = TIER_PRESETS[args.tier]
    logger.info("Run started | tier=%s | params=%s", args.tier, tier_params)
    logger.info("Model path: %s", args.model_path)

    start_run = time.perf_counter()
    plant_df, generator_df, emissions_df = load_tables(
        Path(args.plant_csv),
        Path(args.generator_csv),
        Path(args.emissions_csv),
    )
    eligible_ids = eligible_plant_ids(generator_df, emissions_df)
    logger.info("Eligible plants with NG+SCR: %d", len(eligible_ids))

    if args.sample_size and args.sample_size > 0:
        plant_pool = pd.Series(eligible_ids, dtype="int64")
        selected_ids = (
            plant_pool.sample(n=min(args.sample_size, len(plant_pool)), random_state=args.seed)
            .sort_values()
            .tolist()
        )
    else:
        selected_ids = eligible_ids
    logger.info("Selected plant count: %d | IDs: %s", len(selected_ids), selected_ids)

    llm = Llama(
        model_path=args.model_path,
        n_ctx=tier_params["n_ctx"],
        n_threads=tier_params["n_threads"],
        verbose=False,
    )

    today_year = datetime.now().year
    selected_plants = plant_df[plant_df["eiap_plant_id"].isin(selected_ids)].copy()
    output_rows = []

    for _, plant_row in selected_plants.sort_values("eiap_plant_id").iterrows():
        plant_id = int(plant_row["eiap_plant_id"])
        gens = generator_df[generator_df["eiag_plant_id"].astype("Int64") == plant_id].copy()
        emis = emissions_df[emissions_df["eiaece_plant_id"].astype("Int64") == plant_id].copy()
        bullets = build_plant_bullets(plant_row, gens, emis, today_year)
        prompt = build_prompt(bullets)

        logger.info("Plant %s | prompt begin\n%s\nPlant %s | prompt end", plant_id, prompt, plant_id)
        start_plant = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a power-plant technical analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=tier_params["temperature"],
            top_p=tier_params["top_p"],
            repeat_penalty=tier_params["repeat_penalty"],
            max_tokens=tier_params["max_tokens"],
        )
        elapsed = time.perf_counter() - start_plant

        llm_text = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        logger.info(
            "Plant %s | inference_sec=%.2f | usage=%s | response begin\n%s\nPlant %s | response end",
            plant_id,
            elapsed,
            usage,
            llm_text,
            plant_id,
        )

        output_rows.append(
            {
                "plant_id": plant_id,
                "plant_name": plant_row.get("eiap_plant_name"),
                "tier": args.tier,
                "prompt_text": prompt,
                "llm_output": llm_text,
                "inference_seconds": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )

    out_df = pd.DataFrame(output_rows).sort_values("plant_id")
    out_df.to_csv(csv_path, index=False)

    run_sec = time.perf_counter() - start_run
    logger.info("Run complete | plants=%d | elapsed_sec=%.2f", len(out_df), run_sec)
    logger.info("CSV output: %s", csv_path.resolve())
    logger.info("Log output: %s", log_path.resolve())
    print(f"CSV output: {csv_path.resolve()}")
    print(f"Log output: {log_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
