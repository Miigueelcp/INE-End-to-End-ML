-- Seleccionamos la base de datos
USE portfolio_ine;

-- Creamos la vista que integra los datos de costes laborales y EPA
CREATE OR REPLACE VIEW costes_epa_integrado AS
SELECT 
    c.comunidades_autonomas,
    c.ano,
    c.trimestre,
    c.sectores_actividad,
    c.coste AS tipo_de_coste,
    c.total AS coste_en_euros,
    e.sexo AS sexo,
    e.tasas AS tipo_de_tasa,
    e.total AS valor_tasa

FROM costes_laborales c 

INNER JOIN epa_tasas e
    ON c.comunidades_autonomas = e.comunidades_autonomas
    AND c.ano = e.ano
    AND c.trimestre = e.trimestre
    
WHERE c.sectores_actividad != 'total_sectores'
  AND c.coste = 'coste_laboral_total';



