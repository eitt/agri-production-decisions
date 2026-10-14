# Ejecución local de mantenimiento, 14 de octubre de 2026

`main.py` usa el runner de `agriopt.maintenance`; acepta `--config`, `--run-id`
y `--workers`. Se ejecuta con Python 3.12 de `.venv`, también cuando se invoca
desde otra carpeta. Registra cada lanzamiento de forma atómica, incluido un
bloqueo previo a crear el experimento o una interrupción del operador.

## Estado comprobado

La copia descargada está en `codex/maintenance-resumable`, commit
`62ff8ad974c17bc322ab0027ecf185cb97181b52`, igual a la rama remota consultada.
El paquete de mantenimiento ya existía. Git estaba limpio al iniciar la revisión;
no se encontró `AGENTS.md` en el proyecto ni en los directorios superiores
consultados. El entorno instalado usa Python 3.12.3 y dispone de PySCIPOpt.

La suite completa pasó: **80 pruebas**. La evidencia automática está en
`outputs/maintenance/verification/pytest.xml`, y el resumen de esta revisión en
[MAINTENANCE_LOCAL_VALIDATION.json](MAINTENANCE_LOCAL_VALIDATION.json).
Las pruebas usan fixtures identificadas como datos sintéticos exclusivamente
para verificación; ninguna se instaló como dato del experimento.

Se intentaron tanto `maintenance_smoke.yaml` como el comando local solicitado.
Ambos terminaron con código 1 antes de ejecutar tareas porque faltan los cuatro
datos calibrados. El piloto empírico **no está completado** y el lote de 576
tareas **no está activo**. No hay resultados finales publicados.

La búsqueda comprobó el proyecto y `C:\Users\LEONA\Downloads`, incluida su
configuración de carpeta de Windows. No apareció el archivo requerido. También
se revisaron los ZIP de calibración y del proyecto encontrados en Downloads;
no se utilizaron como reemplazo del paquete de mantenimiento.

## Archivo requerido

Coloque exactamente este paquete en:

```text
C:\Users\LEONA\Documents\GitHub\agri-production-decisions\Datos_mantenimiento_calibrados.zip
```

Debe contener una sola copia de cada archivo:

- `productividad_actualizada.csv`
- `prices_weekly_model_prefix_observed.csv`
- `price_forecast_104w.csv`
- `forecast_provenance.json`

Al repetir el comando, el runner valida el contenido y los hashes SHA256 del
histórico y del pronóstico antes de importar los cuatro archivos en
`data/maintenance/`. No genera pronósticos, no rellena sus precios y no reemplaza
archivos existentes con bytes diferentes. Conserva los hashes del ZIP y de los
cuatro archivos en `data/maintenance/bundle_import.json`.

## Piloto y lote

Desde la raíz del proyecto, ejecute primero el piloto. Si termina correctamente,
inicie el lote:

```powershell
.\.venv\Scripts\python.exe main.py --config configs/maintenance_smoke.yaml --run-id mantenimiento_02_piloto --workers 1
.\.venv\Scripts\python.exe -m agriopt maintenance status --run mantenimiento_02_piloto
.\.venv\Scripts\python.exe main.py --config configs/maintenance_local.yaml --run-id mantenimiento_02 --workers 2
```

El YAML local conserva ocho instancias, seis métodos y doce semillas. Las
heurísticas conservan población 40 y 150 generaciones; cada subproblema epsilon
exacto conserva su límite configurado de 120 segundos, memoria de 512 MB y un
thread. `--workers 2` sustituye el número de procesos para esta ejecución.
No se agregó un límite global vinculado a la sesión del agente.

## Consultar el progreso

```powershell
.\.venv\Scripts\python.exe -m agriopt maintenance status --run mantenimiento_02
Get-Content .\outputs\maintenance\mantenimiento_02\progress.json
Import-Csv .\outputs\maintenance\mantenimiento_02\run_index.csv | Format-Table instance,method,seed,status,generation,evaluations,elapsed_s
Get-ChildItem .\outputs\maintenance\mantenimiento_02\launches\*.json | Sort-Object Name | Select-Object -Last 1 | Get-Content
```

El comando `status` también informa bloqueos de datos cuando todavía no existe
`manifest.json`. Antes de superar ese bloqueo solo existe `launches/`;
`progress.json` y `run_index.csv` se crean al preparar el experimento validado.
El índice y los reportes preliminares se refrescan cada 30 segundos, más el
tiempo de escritura del reporte, incluso durante una fase larga de SCIP con un
solo worker. Los reportes usan la última frontera comprometida, no una generación
o un subproblema todavía sin terminar.

Los avances se guardan bajo `outputs/maintenance/mantenimiento_02/`:

| Archivo o carpeta | Contenido |
|---|---|
| `launches/` | Lanzamientos, bloqueos e interrupciones del operador |
| `manifest.json` | Configuración, datos, código, entorno y procesos fijados por hashes |
| `progress.json`, `run_index.csv` | Conteos y estado de las tareas |
| `tasks/<instancia>__<método>__<semilla>/checkpoint.pkl` | Estado científico reanudable |
| `tasks/.../partial.json`, `status.json` | Último avance comprometido |
| `tasks/.../result.json`, `completion.json` | Resultado auditado y recibo de integridad |
| `preliminary/` | Frentes, hipervolumen e historial, costos, cultivo, mantenimiento y logs del solver |
| `final/` | Reportes y figuras tras completar y auditar todas las tareas |
| `invalidated/` | Archivos anteriores conservados cuando requieren reparación |

Los nombres de los CSV son `fronts.csv`, `metrics.csv`, `summary.csv`,
`hypervolume_history.csv`, `decisions.csv`, `cultivation.csv`, `maintenance.csv`
y `solver_logs.csv`. Los logs exactos incluyen estado, gap, soluciones, nodos,
cotas, tiempo del solver e identidad del subproblema epsilon.

## Reanudar

Repita exactamente:

```powershell
.\.venv\Scripts\python.exe main.py --config configs/maintenance_local.yaml --run-id mantenimiento_02 --workers 2
```

Las tareas con recibos válidos no se repiten. Datos, configuración, código,
entorno o número efectivo de procesos incompatibles requieren otro `run-id`.
Las heurísticas restauran población, archivo Pareto, generación, ambos RNG,
ejemplos de entrenamiento, modelo del surrogate y presupuesto consumido. El
exacto restaura incumbentes y fases epsilon terminadas; **reconstruye el árbol
de SCIP**. Una parada abrupta puede perder trabajo desde el último checkpoint.
Se registra al recuperar una tarea cuyo worker anterior dejó de existir.

La versión de checkpoint es 2; checkpoints de versiones anteriores se rechazan
conservando sus archivos. Los resultados inválidos se archivan antes de reparar
la tarea. Los reportes finales invalidados conservan su contenido y reciben un
archivo separado `invalidation.json`.

Se mantiene el modelo de parcelas completas, intervención en cada octava semana
ociosa, costo supuesto de COP 250.000/ha/intervención y `max_idle_weeks` opcional.
Las restricciones de siembra, maduración, preparación y descanso se conservan.

Para repetir la verificación del código:

```powershell
.\.venv\Scripts\python.exe -m pytest --junitxml=outputs/maintenance/verification/pytest.xml
```
