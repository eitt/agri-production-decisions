
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_FILES = {
    "exact": {
        "front": "exact_front_scip.csv",
        "events": "exact_events_scip.csv",
    },
    "nsga2": {
        "front": "nsga2_front.csv",
        "events": "nsga2_events.csv",
    },
    "nsga2_qp": {
        "front": "nsga2_qp_front.csv",
        "events": "nsga2_qp_events.csv",
    },
    "hnsga2": {
        "front": "hnsga2_front.csv",
        "events": "hnsga2_events.csv",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_run(run: str) -> Path:
    root = repo_root()

    if run.lower() == "latest":
        latest = root / "results" / "LATEST"
        if not latest.exists():
            raise FileNotFoundError("No existe results/LATEST.")
        run = latest.read_text(encoding="utf-8").strip()

    p = Path(run)

    if p.is_absolute():
        return p

    if p.parts and p.parts[0] == "results":
        return root / p

    return root / "results" / p


def choose_solution(
    front: pd.DataFrame,
    criterion: str,
    solution_id: int | None,
) -> int:

    front = front.dropna(subset=["income_mcop", "risk"]).copy()

    if front.empty:
        raise ValueError("El frente no contiene soluciones vÃ¡lidas.")

    if criterion == "id":
        if solution_id is None:
            solution_id = int(front["solution_id"].iloc[0])

        if solution_id not in set(front["solution_id"].astype(int)):
            raise ValueError(
                f"solution_id={solution_id} no existe. "
                f"Disponibles: {sorted(front['solution_id'].astype(int).tolist())}"
            )
        return int(solution_id)

    if criterion == "max-income":
        row = front.loc[front["income_mcop"].idxmax()]
        return int(row["solution_id"])

    if criterion == "min-risk":
        row = front.loc[front["risk"].idxmin()]
        return int(row["solution_id"])

    if criterion == "knee":
        # Ideal aproximado: mÃ¡ximo ingreso y mÃ­nimo riesgo.
        income = front["income_mcop"].astype(float)
        risk = front["risk"].astype(float)

        income_span = max(income.max() - income.min(), 1e-12)
        risk_span = max(risk.max() - risk.min(), 1e-12)

        # 0 es ideal en ambas dimensiones.
        d_income = (income.max() - income) / income_span
        d_risk = (risk - risk.min()) / risk_span

        distance = (d_income**2 + d_risk**2) ** 0.5
        row = front.loc[distance.idxmin()]
        return int(row["solution_id"])

    raise ValueError(f"Criterio desconocido: {criterion}")


def plot_gantt(
    run: str,
    instance: str,
    method: str,
    seed: int,
    criterion: str,
    solution_id: int | None,
    show: bool,
):
    root = repo_root()
    run_dir = resolve_run(run)

    if method not in METHOD_FILES:
        raise ValueError(
            f"MÃ©todo no reconocido: {method}. "
            f"Use uno de {list(METHOD_FILES)}"
        )

    method_dir = (
        run_dir
        / instance
        / method
        / f"seed_{seed}"
    )

    front_path = method_dir / METHOD_FILES[method]["front"]
    events_path = method_dir / METHOD_FILES[method]["events"]

    if not front_path.exists():
        raise FileNotFoundError(f"No existe:\n{front_path}")

    if not events_path.exists():
        raise FileNotFoundError(f"No existe:\n{events_path}")

    front = pd.read_csv(front_path)
    events = pd.read_csv(events_path)

    sid = choose_solution(
        front=front,
        criterion=criterion,
        solution_id=solution_id,
    )

    sol_front = front[
        front["solution_id"].astype(int) == sid
    ].iloc[0]

    sol = events[
        events["solution_id"].astype(int) == sid
    ].copy()

    if sol.empty:
        raise ValueError(
            f"La soluciÃ³n {sid} no contiene eventos agrÃ­colas."
        )

    # Campos esenciales
    sol["lot"] = pd.to_numeric(sol["lot"], errors="raise").astype(int)
    sol["sow"] = pd.to_numeric(sol["sow"], errors="raise").astype(int)
    sol["harvest"] = pd.to_numeric(sol["harvest"], errors="raise").astype(int)
    sol["duration"] = sol["harvest"] - sol["sow"]

    # Algunos eventos producidos directamente por NSGA-II no contienen
    # farm / municipality / yield_kg_m2. No son necesarios para el Gantt.
    if "farm" not in sol.columns:
        sol["farm"] = pd.NA

    sol["farm"] = pd.to_numeric(sol["farm"], errors="coerce").astype("Int64")

    if "municipality" not in sol.columns:
        sol["municipality"] = pd.NA

    sol["municipality"] = sol["municipality"].fillna("")

    if "yield_kg_m2" not in sol.columns:
        sol["yield_kg_m2"] = pd.NA

    sol = sol.sort_values(
        ["farm", "lot", "sow", "harvest"]
    )

    # Para la representaciÃ³n temporal, el lote es la unidad fundamental.
    # farm y municipality se muestran Ãºnicamente cuando estÃ¡n disponibles.
    lots = (
        sol[["lot"]]
        .drop_duplicates()
        .sort_values("lot")
        .reset_index(drop=True)
    )

    lot_info = (
        sol.groupby("lot", as_index=False)
        .agg(
            farm=("farm", "first"),
            municipality=("municipality", "first"),
        )
    )

    lots = lots.merge(lot_info, on="lot", how="left")

    lot_position = {
        int(row.lot): i
        for i, row in lots.iterrows()
    }

    crops = sorted(sol["crop_name"].dropna().unique())

    cmap = plt.get_cmap("tab20")
    crop_color = {
        crop: cmap(i % 20)
        for i, crop in enumerate(crops)
    }

    # Altura adaptable al nÃºmero de lotes.
    fig_height = max(4.5, 0.62 * len(lots) + 2.2)

    fig, ax = plt.subplots(
        figsize=(13, fig_height)
    )

    bar_height = 0.62

    for _, row in sol.iterrows():

        y = lot_position[int(row["lot"])]

        crop = str(row["crop_name"])
        sow = int(row["sow"])
        harvest = int(row["harvest"])
        duration = harvest - sow

        ax.barh(
            y=y,
            width=duration,
            left=sow,
            height=bar_height,
            color=crop_color[crop],
            edgecolor="black",
            linewidth=0.5,
        )

        area = float(row["area_m2"])
        production = float(row.get("production_kg", float("nan")))

        label = f"{crop}\n{area:,.0f} mÂ²"

        # Solo colocar texto dentro si la barra tiene espacio.
        if duration >= 5:
            ax.text(
                sow + duration / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
            )

        # Marcas de siembra y cosecha.
        ax.plot(
            sow,
            y,
            marker="|",
            markersize=13,
            color="black",
        )
        ax.plot(
            harvest,
            y,
            marker="|",
            markersize=13,
            color="black",
        )

    labels = []

    for _, row in lots.iterrows():
        parts = []

        if pd.notna(row["farm"]):
            parts.append(f"F{int(row['farm']) + 1}")

        parts.append(f"L{int(row['lot']) + 1}")

        municipality = str(row.get("municipality", "")).strip()
        if municipality and municipality.lower() != "nan":
            parts.append(municipality)clear

        labels.append(" Â· ".join(parts))

    ax.set_yticks(range(len(lots)))
    ax.set_yticklabels(labels)

    ax.invert_yaxis()

    max_week = int(sol["harvest"].max())

    # CuadrÃ­cula temporal cada 4 semanas.
    ticks = list(range(0, max_week + 5, 4))
    ax.set_xticks(ticks)

    ax.set_xlabel("Semana del horizonte de planificaciÃ³n")
    ax.set_ylabel("Finca Â· lote Â· municipio")

    income = float(sol_front["income_mcop"])
    risk = float(sol_front["risk"])

    title = (
        f"SecuenciaciÃ³n agrÃ­cola â€” {instance}\n"
        f"{method} Â· seed {seed} Â· soluciÃ³n {sid} Â· "
        f"Ingreso = {income:,.3f} MCOP Â· Riesgo = {risk:,.4g}"
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.35,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Leyenda por cultivo.
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=crop_color[crop],
            edgecolor="black",
            linewidth=0.4,
            label=crop,
        )
        for crop in crops
    ]

    ax.legend(
        handles=handles,
        title="Cultivo",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )

    fig.text(
        0.01,
        0.01,
        "La barra representa el intervalo siembraâ€“cosecha. "
        "El texto indica cultivo y Ã¡rea asignada.",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.035, 0.86, 1))

    out_dir = (
        root
        / "artifacts"
        / run_dir.name
        / "figures"
        / "schedules"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"gantt_{instance}_{method}_"
        f"seed_{seed}_solution_{sid}"
    )

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    print()
    print("Gantt generado")
    print("----------------")
    print(f"Run:       {run_dir.name}")
    print(f"Instancia: {instance}")
    print(f"MÃ©todo:    {method}")
    print(f"Seed:      {seed}")
    print(f"SoluciÃ³n:  {sid}")
    print(f"Ingreso:   {income:.6f} MCOP")
    print(f"Riesgo:    {risk:.8g}")
    print(f"Eventos:   {len(sol)}")
    print()
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualiza como Gantt la secuenciaciÃ³n agrÃ­cola "
            "de una soluciÃ³n del frente de Pareto."
        )
    )

    parser.add_argument(
        "--run",
        default="latest",
        help="Run ID o 'latest'.",
    )

    parser.add_argument(
        "--instance",
        default="I01_micro",
    )

    parser.add_argument(
        "--method",
        choices=list(METHOD_FILES),
        default="hnsga2",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--criterion",
        choices=[
            "id",
            "max-income",
            "min-risk",
            "knee",
        ],
        default="knee",
        help=(
            "CÃ³mo escoger el punto del frente. "
            "'knee' selecciona un compromiso ingreso-riesgo."
        ),
    )

    parser.add_argument(
        "--solution-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="AdemÃ¡s de guardar PNG/PDF, abre la figura.",
    )

    args = parser.parse_args()

    plot_gantt(
        run=args.run,
        instance=args.instance,
        method=args.method,
        seed=args.seed,
        criterion=args.criterion,
        solution_id=args.solution_id,
        show=args.show,
    )


if __name__ == "__main__":
    main()
