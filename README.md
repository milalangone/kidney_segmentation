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

[Drive: Data set Normal-Cyst-Tumor and Stone](https://drive.google.com/drive/folders/1WzF3_uMtMcyLssO3RGUwCQIadztsjEks?usp=drive_link)

### Data Pipeline

![data_pipeline](https://github.com/milalangone/pib_ct_kidney_segmentation/assets/89553721/e3d089dd-d754-4463-bfd4-3e2bba704402)

### Estructura de los datos

Además de contar con las imágenes per se, también se cuenta con un .csv que contiene los siguientes campos:

| image_id | path | diag | target | class |
| --- | --- | --- | --- | --- |

## GUI
### Procesamiento de las imágenes

Cuando se carga una imagen a la GUI, el proceso por el que pasa es el siguiente:

![pako_eNqFkF1LwzAUhv9KCAgVVoZe9kJw_Vr9AHF6Y9KLQ3vaFZukpCk6tv1302YdKoK5Sd5znve8Sfa0UCXSgFat-ii2oA15eOaS2HXLlpmAGpc58f0bsmLpa5a71mqqHB5BDtASzmWPLRamUfJAQo_dN6XEHbnKL__HozN-PeLOEI58zNiTRr_TqsC-b2Sdn9KjaVzsRDyJhLEN1gKlg](https://github.com/milalangone/pib_ct_kidney_segmentation/assets/89553721/7d2329cd-8b4c-4f6c-9816-db9039bd4c38)


## Bibliografía

- Islam MN, Hasan M, Hossain M, Alam M, Rabiul G, Uddin MZ, Soylu A. Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography. Scientific Reports. 2022 Jul 6;12(1):1-4.
