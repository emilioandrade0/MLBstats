# StatsMLB

Dashboard en español para consultar los juegos de hoy y estudiar cómo responden los equipos después de una serie de tres juegos.

## Qué incluye

- Interfaz organizada en siete pestañas: Jornada, Comparador, Rendimiento, Auditorías, Equipos y series, Momios y Método.
- Subpestañas para las ocho auditorías, rankings/escenarios/barridas y calendario/combinaciones. Los filtros y el modelo activo se conservan al cambiar de vista.
- Navegación por teclado con flechas, Inicio y Fin, enlaces directos a las secciones e historial del navegador. Tablas desplazables y diseño adaptable a móvil.
- Calendario del día desde MLB StatsAPI, con respaldo local claramente etiquetado.
- Rankings de equipos que barren, equipos vulnerables, respuesta tras barrer y rebote tras ser barridos.
- Escenarios donde se enfrentan dos equipos que barrieron, dos que fueron barridos y un barrido 0-3 contra un perdedor 1-2.
- Estimación explicable de victoria con controles para activar o quitar localía, fuerza de temporada, forma L10, diferencial de carreras y serie previa.
- Calendario walk-forward mensual con rendimiento por día, detalle juego por juego, resultado real y fecha máxima utilizada para entrenar cada corte.
- Los diez controles de factores son compartidos: al desmarcar uno se recalculan de inmediato el calendario walk-forward, sus aciertos, colores, temporadas y bandas de confianza.
- Comparador de las 1,023 combinaciones no vacías de factores, ordenado por acierto walk-forward total y desglosado por temporada; cualquier combinación se puede aplicar con un clic.
- Fatiga del lineup (prueba): usa titularidades y apariciones de plato de los bateadores confirmados en ventanas de 3/7 días y descanso menor a 30 horas. Está apagada por defecto y solo se aplica con cobertura suficiente en ambos lineups.
- Las pruebas de mejores jugadores, rotación del coach, calidad de la rotación y fatiga del lineup permanecen apagadas por defecto y requieren datos pregame suficientes.
- Preparador de picks para Telegram por partido: usa el mejor moneyline disponible, permite elegir equipo y número de pick, muestra la vista previa exacta y exige confirmación manual antes de publicar.
- Seis análisis adicionales preparados como hoja de ruta: abridor, bullpen, lineup, descanso/viaje, parque/clima y cara a cara.

## Abrir

Ejecuta `ABRIR StatsMLB.bat`. El iniciador espera a que el servidor esté listo y abre `http://127.0.0.1:4173`.

## Actualizar el histórico

Ejecuta `ACTUALIZAR HISTORICO.bat` después de actualizar los archivos de STRIKECAST. El proceso solo lee la carpeta `data` original, reconstruye los derivados en `work`, recalcula el estimador y compila el sitio.

## Conectar Telegram

1. Crea un bot con `@BotFather` usando `/newbot` y conserva el token en privado.
2. Añade el bot como administrador del canal con permiso para publicar mensajes.
3. Ejecuta `CONFIGURAR TELEGRAM.bat`, completa `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`, guarda el archivo y avisa a Codex que la configuración está lista.

El archivo privado se guarda fuera del repositorio en `%LOCALAPPDATA%\StatsMLB\telegram.env`. El token nunca se incorpora al navegador, a los datos públicos ni al control de versiones.

## Método y procedencia

La pestaña **Fallos y riesgo** lee el histórico publicado en el mismo sitio (sin caché) y muestra su corte real. Compara aciertos y fallos de la máscara fija 45, sin seleccionar retrospectivamente la mejor combinación. Incluye señales con/sin presencia, simulación de abstención con cobertura y aciertos descartados, estabilidad mensual, partidos y descarga JSON. El corte de diseño es 2025-12-31 y la evaluación retrospectiva comienza en 2026; no es una prueba prospectiva ni activa reglas nuevas. En **Comparador**, «Revisar fallos de esta jornada» examina las recomendaciones mostradas, distinguiendo esa reconstrucción del histórico del motor. El registro inmutable de recomendaciones emitidas en vivo todavía es necesario para entrenar un detector específico de fallos LOCK/FUERTE.

Validación del diagnóstico: `node --experimental-strip-types --test scripts/test_failure_analysis.mjs`.

La definición de barrida exige una serie exacta de tres juegos ganada 3-0. El calendario histórico usa un walk-forward expansivo: al iniciar cada mes, el estimador se entrena solo con juegos terminados antes de ese mes y queda congelado hasta el siguiente corte. Entre marzo de 2024 y agosto de 2026 produjo 6,822 predicciones verdaderamente fuera de muestra. La combinación recomendada obtuvo 56.24%; la prueba opcional de calidad de rotación obtuvo 57.45%. Los porcentajes son estimaciones informativas, no garantías.

Cuando se desactiva un factor, el navegador elimina únicamente la contribución de ese grupo dentro del modelo mensual que ya estaba congelado y vuelve a calcular sus predicciones. Esta comparación no reentrena el modelo ni utiliza resultados futuros.

Antes de generar el calendario, `build_data.py` compara el derivado de series contra todos los juegos finales disponibles en `data/processed/games.parquet`. Si falta un solo juego o una fecha queda incompleta, la generación se detiene con un error en lugar de publicar métricas parciales.
