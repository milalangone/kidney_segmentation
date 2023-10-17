# Clasificación de quistes, tumores, y cálculos renales a partir de imágenes de tomografía computarizada

**16.16 - Procesamiento de Imágenes Biomédicas**

Alumnas:
- Mila Langone (61273)
- Francesca Rondinella (61031)
- Lucía Simoncelli (61429)
- Shirley Terrazas (59471)

## Introducción

El siguiente trabajo estará basado en la base de datos [CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone), que será utilizada para clasificar quistes, tumores, cálculos renales e imágenes sin patologías a partir de tomografías computarizadas. Se cuenta con la clasificación de las mismas hecha por un profesional médico, por lo que se realizará una clasificación supervisada con Machine Learning.

## Drive

https://drive.google.com/drive/folders/1WzF3_uMtMcyLssO3RGUwCQIadztsjEks?usp=drive_link

### Data Pipeline

![data_pipeline](https://github.com/milalangone/pib_ct_kidney_segmentation/assets/89553721/e3d089dd-d754-4463-bfd4-3e2bba704402)

### Estructura de los datos

Además de contar con las imágenes per se, también se cuenta con un .csv que contiene los siguientes campos:

| image_id | path | diag | target | class |
| --- | --- | --- | --- | --- |


## Bibliografía

- Islam MN, Hasan M, Hossain M, Alam M, Rabiul G, Uddin MZ, Soylu A. Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography. Scientific Reports. 2022 Jul 6;12(1):1-4.
