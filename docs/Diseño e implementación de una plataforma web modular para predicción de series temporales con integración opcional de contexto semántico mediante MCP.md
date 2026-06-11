# **Diseño e implementación de una plataforma web modular para predicción de series temporales con integración opcional de contexto semántico mediante MCP**

Design and implementation of a reproducible web-based platform for experimentation and comparative evaluation of time series forecasting models with semantic context integration via large language models and the Model Context Protocol

This project presents the design and implementation of a modular and reproducible web-based platform for managing, executing, and comparing time series forecasting experiments with optional semantic context integration. The system is conceived as the architectural counterpart to a parallel research study evaluating the predictive impact of contextual augmentation, translating experimental workflows into a structured and auditable software environment.

The platform encapsulates core components including data ingestion pipelines, forecasting engines, experiment management services, and a semantic context module within a layered architecture that clearly separates presentation, gateway, application, and persistence layers. This design enables controlled execution of forecasting runs under configurable conditions, including the activation or deactivation of contextual signals obtained through the Model Context Protocol (MCP), without modifying underlying model implementations.

A documented API exposes functionality for dataset versioning, experiment configuration, model execution, and result retrieval, while a web-based dashboard supports interactive comparison of runs and visualization of outputs. The system records configuration parameters, dataset versions, and execution metadata to ensure reproducibility and traceability.

By emphasizing encapsulation, modularity, and separation of concerns, the project demonstrates how sound software engineering principles can transform experimental machine learning workflows into maintainable, extensible, and production-oriented systems suitable for rigorous analytical environments.

## **Resumen ejecutivo**

Este TFG propone el diseño e implementación de una plataforma web profesional para el análisis y predicción de series temporales económicas, permitiendo la integración opcional de contexto semántico derivado de noticias mediante el protocolo Model Context Protocol (MCP).

El sistema permitirá:

* Ingesta y almacenamiento de múltiples bases de datos económicas.

* Gestión automatizada de procesos ETL.

* Selección dinámica de series temporales.

* Elección interactiva del modelo predictivo.

* Activación o desactivación del uso de contexto semántico.

* Visualización comparativa de resultados.

* Persistencia de experimentos y métricas en base de datos.

La arquitectura seguirá principios de ingeniería profesional:

* Backend principal en **FastAPI**

* API Gateway en **Node.js**

* Base de datos relacional para series y experimentos

* Base NoSQL para noticias

* Arquitectura modular orientada a microservicios

* Integración opcional con agentes LLM vía MCP

El objetivo no es únicamente desarrollar una aplicación funcional, sino construir una arquitectura robusta, escalable y reproducible alineada con buenas prácticas de ingeniería de software moderna.

## **Motivación**

En entornos reales de análisis económico y financiero, los procesos de forecasting suelen estar fragmentados:

* Scripts aislados.

* Falta de trazabilidad.

* Escasa reproducibilidad.

* Difícil comparación entre modelos.

* Poca integración con fuentes externas dinámicas.

Además, la integración de agentes LLM y protocolos como MCP requiere una arquitectura controlada y segura que permita activar o desactivar componentes sin romper el sistema.

Existe, por tanto, la oportunidad de diseñar una plataforma:

* Modular.

* Escalable.

* Reproducible.

* Preparada para investigación experimental.

## **Objetivo general**

Diseñar e implementar una plataforma web modular para la gestión, modelado y evaluación de series temporales económicas, permitiendo la integración opcional de contexto semántico mediante MCP dentro de una arquitectura profesional basada en microservicios.

## **Objetivos específicos**

1. Diseñar una arquitectura orientada a servicios que desacople:

   * Ingesta de datos.

   * Procesamiento.

   * Modelado.

   * Visualización.

   * Integración MCP.

2. Implementar un sistema ETL configurable para:

   * Series económicas.

   * Noticias/eventos.

3. Diseñar un modelo de base de datos relacional para:

   * Series temporales.

   * Metadatos.

   * Experimentos.

   * Resultados y métricas.

4. Implementar un gateway en Node.js que:

   * Centralice autenticación.

   * Orqueste llamadas a servicios.

   * Permita versionado de API.

5. Desarrollar un backend en FastAPI que:

   * Exponga endpoints de modelado.

   * Permita selección dinámica de modelo.

   * Active/desactive el uso de MCP.

6. Diseñar una interfaz web interactiva que permita:

   * Seleccionar dataset.

   * Seleccionar horizonte.

   * Elegir modelo.

   * Activar contexto.

   * Visualizar comparativas.

7. Garantizar reproducibilidad mediante:

   * Registro de configuraciones.

   * Versionado de datos.

   * Persistencia de experimentos.

## **Arquitectura propuesta**

### **Visión general**

Arquitectura modular en capas:

Frontend  
 ↓  
 API Gateway (Node.js)  
 ↓  
 Servicios backend (FastAPI)  
 ↓  
 Bases de datos \+ Servicios MCP

### **Componentes principales**

#### **1\. Frontend (React / Next.js)**

Funcionalidades:

* Selector de base de datos

* Selector de serie

* Selector de modelo

* Toggle “Usar contexto MCP”

* Configuración de horizonte

* Visualización gráfica interactiva

* Comparativa C0 vs C1

* Panel de métricas

#### **2\. API Gateway (Node.js)**

Responsabilidades:

* Autenticación y autorización

* Rate limiting

* Logging centralizado

* Enrutamiento a microservicios

* Versionado de endpoints

Permite separar lógica de acceso de lógica de negocio.

#### **3\. Backend principal (FastAPI)**

Servicios:

* Servicio de modelado

* Servicio de experimentos

* Servicio de ETL

* Servicio MCP client

Endpoints ejemplo:

* /datasets

* /models

* /forecast

* /experiments

* /metrics

#### **4\. Bases de datos**

### **Base relacional (PostgreSQL)**

Tablas:

* datasets

* series

* observations

* models

* experiments

* predictions

* metrics

### **Base NoSQL (MongoDB o Elastic)**

* Noticias

* Eventos

* Resultados estructurados del agente

* Logs de extracción

#### **5\. Integración MCP**

Componente desacoplado:

* Cliente MCP dentro de backend

* Conexión a servidor MCP

* Acceso a:

  * Recursos (noticias)

  * Tools (extracción estructurada)

  * Prompts versionados

Debe poder activarse/desactivarse sin modificar la lógica del modelo.

## **Pipeline del sistema**

1. Ingesta de datos (ETL)

2. Persistencia en base de datos

3. Selección por usuario

4. Construcción del experimento

5. Activación opcional del agente MCP

6. Generación de variables exógenas

7. Ejecución del modelo

8. Persistencia de resultados

9. Visualización interactiva

## **Características técnicas clave**

* Arquitectura desacoplada

* Modularidad

* Control de dependencias

* Logging estructurado

* Manejo de errores

* Validación de datos

* Control de timestamps para evitar fuga temporal

* Persistencia de configuraciones de ejecución

## **Reproducibilidad**

Cada experimento guardará:

* Dataset usado

* Ventana temporal

* Modelo

* Hiperparámetros

* Uso o no de MCP

* Configuración del agente

* Métricas obtenidas

* Timestamp de ejecución

Esto convierte la plataforma en una herramienta de investigación reproducible.

## **Escalabilidad**

Posibilidades futuras:

* Dockerización completa

* Despliegue en cloud

* Separación real de microservicios

* Cola de tareas asíncronas (Celery / Redis)

* Paralelización de experimentos

## **Seguridad**

* Autenticación JWT

* Separación de roles

* Protección de endpoints sensibles

* Aislamiento de credenciales MCP

* Control de acceso a bases de datos

## **Riesgos y limitaciones**

* Complejidad arquitectónica excesiva para un TFG.

* Gestión de estados entre servicios.

* Costes asociados a llamadas LLM.

* Latencia en ejecución de modelos pesados.

* Sobredimensionamiento si no se acota bien el alcance.

## **Estructura propuesta de memoria**

1. Introducción

2. Requisitos del sistema

3. Arquitectura propuesta

4. Diseño de base de datos

5. Diseño de APIs

6. Implementación

7. Integración MCP

8. Pruebas y validación

9. Evaluación técnica

10. Limitaciones y mejoras futuras

11. Conclusiones

