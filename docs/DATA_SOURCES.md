# Data sources and lineage

## Wholesale prices: DANE SIPSA-P

The live price pipeline uses the official DANE **Sistema de Información de Precios y Abastecimiento del Sector Agropecuario (SIPSA)** wholesale-price publications. DANE currently provides annual historical-series workbooks and a SOAP web service for consolidated wholesale-price queries.

Official pages used by the code:

- SIPSA portal: https://www.dane.gov.co/index.php/servicios-al-ciudadano/servicios-informacion/sipsa
- Monthly wholesale-price page and historical series: https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/sistema-de-informacion-de-precios-sipsa/mayoristas-boletin-mensual-1
- SIPSA SOAP WSDL: http://appweb.dane.gov.co/sipsaWS/SrvSipsaUpraBeanService?WSDL

`agriopt data update` caches the official XLSX files in `data/raw/sipsa_prices/`, normalizes their changing column names, filters Bucaramanga/Centroabastos, maps DANE product labels to the canonical optimization labels in `configs/products.yaml`, and writes a weekly matrix to `data/processed/prices_weekly.csv`.

The mapping audit is stored in `data/interim/price_mapping_coverage.csv`; this file should be reviewed before publication because DANE product labels and market coverage can change.

## Food inflows: SIPSA-A

SIPSA-A reports food quantities entering wholesale markets. These quantities are not identical to final consumer demand, so this project describes the derived constraint as a **market-absorption/throughput proxy** rather than observed demand.

The code can query the DANE SOAP operation `promedioAbasSipsaMesMadr`, stores the untouched payload under `data/raw/sipsa_soap/`, and attempts schema discovery before aggregating the four sales groups used by the model. The processed planning cap uses month-of-year median wholesale inflows and extends that seasonal profile over the configured forecast horizon; it is therefore a transparent market-throughput proxy rather than a consumer-demand forecast. If DANE changes the SOAP schema, the pipeline fails with the available fields while preserving the raw payload for parser adjustment.

An alternative public archive is available through Datos Abiertos Colombia under the SIPSA-A dataset; this repository does not hard-code a downloadable attachment identifier because the catalog attachments are periodically updated.

## Productivity

`data/seed/productividad_actualizada.csv` is the current study productivity table supplied with the project. The optimization code estimates heterogeneous yields hierarchically across product, municipality-product, product-month, and municipality-product-month cells, using shrinkage toward broader groups when local support is sparse.

Before public repository release, confirm the redistribution status and citation requirements for this file.

## Data provenance

Every live update creates `data/processed/data_provenance.json`, containing source years, market filters, environment information, input hashes, and official source endpoints. Downloaded source files are intentionally not tracked by Git.
