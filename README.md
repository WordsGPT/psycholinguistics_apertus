# Psycholinguistics Experiments with Language Models

Un sistema completo y robusto para ejecutar experimentos psicolingüísticos utilizando modelos de lenguaje grandes (LLMs). Este proyecto está específicamente diseñado para evaluar la **familiaridad de palabras** utilizando el modelo **Apertus-8B-Instruct**.

## Descripción del Proyecto

Este repositorio implementa un pipeline automatizado para:

- **Experimentos de Familiaridad**: Evalúa qué tan familiares son las palabras en una escala de 1-7
- **Procesamiento por Lotes**: Maneja grandes datasets de forma eficiente
- **Análisis de Probabilidades**: Captura log-probabilidades detalladas para cada respuesta
- **Robustez**: Sistema de recuperación automática y validación de datos
- **Progreso Visual**: Barras de progreso en tiempo real y reportes detallados

## Inicio Rápido

### 1. Configurar credenciales

Para modelos de Hugging Face privados o que requieren autorización, crea un archivo `api.env` como el del ejemplo `api_example.env`:

```env
HUGGINGFACE_TOKEN=tu_token_aqui
```

### 2. Configurar experimento

Edita las variables en la segunda celda de `main.ipynb`:

```python
experiment_name = "<EXPERIMENT_NAME>"       	# Nombre del experimento
experiment_path = "<EXPERIMENT_PATH>"    	# Carpeta del experimento
```

### 3. Ejecutar experimento

Abre `main.ipynb` y ejecuta las celdas secuencialmente:

1. **Instalar dependencias** - Instala paquetes necesarios
2. **Login Hugging Face** - Autenticación opcional para modelos privados
3. **Cargar modelo** - Carga Apertus-8B-Instruct (8B parámetros)
4. **Preparar experimento** - Genera prompts desde el dataset Excel
5. **Ejecutar experimento** - Procesa palabras y guarda resultados

## Arquitectura del Sistema

### Estructura del Proyecto

```
psycholinguistics_german/
├── main.ipynb                      	#  Notebook principal de ejecución
├── requirements.txt                 	#  Dependencias del proyecto
├── api.env                         	#  Credenciales
│
├── familiarity_german/              	#  Carpeta del experimento
│   ├── config.yaml                  		#  Configuración del experimento
│   ├── data/                        		#  Datos de entrada
│   │   └── datasets.xlsx
│   ├── prompts/                     		#  Templates de prompts
│   │   └── prompt_template.txt  
│   ├── batches/                     		#  Archivos JSONL generados
│   │   └── experiment_name.jsonl     			# Prompts preparados para procesamiento
│   └── outputs/                     		#  Resultados finales
│       └── experiment_name.xlsx      			# Tabla con respuestas y log-probabilidades
│
└── scripts/                         	#  Módulos de procesamiento
    ├── utils.py                     		#  Utilidades (YAML, Excel, CSV)
    ├── prepare_experiment.py        		#  Generación de batches JSONL
    └── execute_experiment.py         		#  Motor de ejecución principal
```
