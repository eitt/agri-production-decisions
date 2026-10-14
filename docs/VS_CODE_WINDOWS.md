# Ejecutar el modelo con mantenimiento en VS Code

Abra la carpeta raíz del repositorio en VS Code. Use Python 3.12 de 64 bits y
la extensión Python. Los comandos siguientes usan la terminal PowerShell de
VS Code y el intérprete del entorno virtual directamente; no necesitan activar
el entorno con un script de PowerShell.

## Instalación

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[scip,dev]"
.\.venv\Scripts\python.exe -m agriopt doctor
```

En `Ctrl+Shift+P` > `Python: Select Interpreter`, seleccione
`.venv\Scripts\python.exe`. El método exacto necesita PySCIPOpt; un error de
instalación o de solver se registra y debe resolverse antes de comparar todos
los métodos. Descargue `Datos_mantenimiento_calibrados.zip` y colóquelo en la
carpeta raíz del proyecto. El runner valida los cuatro archivos y la procedencia
del pronóstico en una carpeta temporal antes de importarlos en `data/maintenance/`.
Rechaza reemplazar archivos existentes con bytes diferentes. Los datos quedan
excluidos de Git; la configuración verifica sus hashes antes de usarlos.

La entrada sencilla es `main.py`; consulte los comandos y la evidencia local en
[MAINTENANCE_LOCAL_EXECUTION.md](MAINTENANCE_LOCAL_EXECUTION.md).

## Prueba inicial

```powershell
.\.venv\Scripts\python.exe -m agriopt maintenance run --config configs/maintenance_smoke.yaml --run-id mantenimiento_prueba --workers 1
.\.venv\Scripts\python.exe -m agriopt maintenance status --run mantenimiento_prueba
.\.venv\Scripts\python.exe -m agriopt maintenance report --run mantenimiento_prueba
```

Esta prueba comprueba las seis estrategias en una instancia pequeña; no sirve
para concluir que una estrategia es superior.

## Lote con generaciones fijas

```powershell
.\.venv\Scripts\python.exe -m agriopt maintenance run --config configs/maintenance_local.yaml --run-id mantenimiento_local_01 --workers 2
```

Las heurísticas completan las generaciones configuradas. El exacto usa límites
explícitos por subproblema epsilon. Estos presupuestos no representan la misma
cantidad de trabajo entre algoritmos; el reporte los identifica.

## Comparación con presupuesto de tiempo

```powershell
.\.venv\Scripts\python.exe -m agriopt maintenance run --config configs/maintenance_publication.yaml --run-id mantenimiento_tiempo_01 --workers 2
```

Incluye ocho instancias, seis métodos y doce semillas: 576 tareas. Los límites
son 20, 20, 40, 20, 40, 60, 80 y 120 segundos de búsqueda por instancia. La
carga de datos y la inicialización común quedan fuera del presupuesto. El
guardado y las actualizaciones del reporte se contabilizan por separado. El
protocolo nuevo documenta estas reglas y no se combina con resultados de
versiones anteriores sin comprobar su comparabilidad.

Dos procesos son un punto de partida para un computador personal. El consumo
del exacto depende del tamaño de la instancia; ajuste `experiment.exact.memory_limit_mb` en el YAML
antes de crear el lote. Una tarea con límite de memoria conserva sus soluciones
y registra el estado y el gap del solver; no constituye una certificación del frente completo.

## Reanudar y consultar

Repita **el mismo comando, con el mismo YAML y el mismo `run-id`**. Las tareas
completas se validan y se omiten. Las heurísticas recuperan población, frente,
estado aleatorio y aprendizaje desde la última generación guardada. El exacto
recupera subproblemas terminados e incumbentes; al reconstruir SCIP no recupera
su árbol interno de búsqueda.

```powershell
.\.venv\Scripts\python.exe -m agriopt maintenance status --run mantenimiento_local_01
.\.venv\Scripts\python.exe -m agriopt maintenance report --run mantenimiento_local_01
```

Los archivos de estado, resultados y checkpoints se escriben de forma atómica.
Un cierre abrupto puede perder el trabajo posterior al último checkpoint. Las
reanudaciones quedan registradas; no se presenta una tarea reiniciada como una
trayectoria de tiempo ininterrumpida. Los checkpoints son archivos pickle
locales: utilice únicamente los generados por su propia ejecución.

Los resultados se guardan en `outputs/maintenance/<run-id>/`. Allí encontrará
`tasks/` con los checkpoints, `run_index.csv`, `preliminary/` con CSV actualizados
y `final/` con las tablas y figuras PDF/SVG/PNG al completar todas las tareas.

Cambiar datos, parámetros, número de procesos o código exige otro `run-id`; la reanudación rechaza
mezclas. Use identificadores explícitos y evite ejecutar dos procesos sobre el
mismo lote. Ejecute desde la carpeta raíz. Puede cerrar la conversación de
ChatGPT mientras el proceso continúa en su computador. Mantenga el computador
encendido y la terminal abierta; cerrar la terminal puede detener el proceso.

## Sensibilidad del mantenimiento y del límite sin cultivar

```powershell
.\.venv\Scripts\python.exe -m agriopt maintenance scenarios --config configs/maintenance_publication.yaml --run-id mantenimiento_sensibilidad_01
```

La cuadrícula principal tiene 192 casos: intervalos de 4/8/12 semanas,
costos de 0/100.000/250.000/500.000 COP por ha e intervención, y límite de
inactividad ausente o de 26 semanas. Añade 32 diagnósticos de límites estrictos
de 4/8/12/26 semanas. La programación dinámica optimiza el extremo económico
y evalúa su riesgo; no optimiza un frente multiobjetivo.

En el YAML puede cambiar:

```yaml
maintenance:
  interval_weeks: 8
  cost_cop_ha_pass: 250000
  max_idle_weeks: null
```

`max_idle_weeks: 26` limita semanas consecutivas sin cultivo, incluidos los
extremos del horizonte. El mantenimiento no reinicia ese contador. La opción
puede hacer inviable una instancia por sus ventanas reales de siembra.

El costo es un supuesto de sensibilidad, no una tarifa observada. El primer
objetivo es ingreso bruto menos mantenimiento; no es utilidad porque no incluye
todos los costos de producción. Consulte [la formulación](MAINTENANCE_MODEL.md).

## Verificación

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_maintenance_model.py tests/test_maintenance_resume.py tests/test_maintenance_runner.py
```

Los reportes incrementales son preliminares mientras falten tareas. Utilice el
reporte final y sus auditorías para redactar conclusiones científicas.
