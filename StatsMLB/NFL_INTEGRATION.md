# Integración NFL

La ruta `/nfl` comparte el diseño y selector deportivo de MLB, con una variante azul oscuro. Incluye Jornada, Historial, Comparador, Rendimiento y Laboratorio (ROI/patrones originales). La aplicación NFL original no se modifica. Los pequeños helpers compartidos ahora están en `lib/nfl/`: el despliegue web no requiere subir toda la aplicación NFL original.

## Datos actuales en la web

`GET /api/nfl/live?date=2026-09-20&season=2026&week=2&phase=reg` consulta el scoreboard público NFL de Action Network desde el servidor. La web identifica la semana más cercana a la fecha en el calendario exportado (máximo siete días) y envía temporada, semana y fase explícitas. La fuente ignora `date` para NFL: no debe usarse como único filtro. `phase=post` admite rondas 1–4; se convierten a la numeración histórica de la app. Si la fuente devuelve otra semana, se rechaza la respuesta en vez de mezclarla. La web consulta al abrir, al cambiar fecha y cada 60 segundos mientras está visible; el servidor reutiliza consultas de menos de 45 segundos. El botón Actualizar datos NFL solicita nuevamente los datos (respetando esa caché corta). Fechas fuera del calendario exportado requieren actualizar ese calendario; se muestra un aviso en lugar de adivinar una semana.

La fecha consultada permite recuperar semanas anteriores. Los juegos observados se guardan en tablas D1 separadas `nfl_live_games` y `nfl_live_checks`, creadas automáticamente en el binding `DB` existente. Se recupera lo guardado si falla la fuente, con aviso explícito. Sin D1, la consulta funciona pero avisa que no se guardará. No hay actualizaciones automáticas con todas las páginas cerradas; al abrir se consulta de nuevo. No se promete tiempo real instantáneo ni disponibilidad contractual de este endpoint público.

Los marcadores y horarios se combinan con el snapshot mediante equipos canónicos, temporada y proximidad de fecha (o ID de fuente ya conocido), sin invertir local/visitante. Un resultado final no vuelve a programado por una respuesta atrasada. Los momios actuales se muestran separados con casa y hora: nunca reemplazan las líneas con las que se evaluaron picks históricos. La consulta no reentrena ni genera nuevas probabilidades, y los filtros históricos no se recalculan en vivo.

## GitHub y Cloudflare

Para publicar: subir los cambios de `StatsMLB/app/`, `StatsMLB/lib/`, `StatsMLB/db/nfl-live.ts` y los JSON iniciales `StatsMLB/public/data/nfl/`, junto con los cambios de configuración de la integración. El workflow existente `actualizar-historico` ya compila StatsMLB y despliega Cloudflare; no necesita un nuevo secret de Action Network ni un servicio Python adicional para consultar marcadores. No se hizo push ni despliegue desde esta tarea. Las credenciales y `.dev.vars` nunca deben subirse a GitHub.

El Worker necesita el binding D1 `DB` ya utilizado por MLB. Para Telegram, configurar `NFL_TELEGRAM_CHAT_ID=@StrikeCastNFL` en el entorno del Worker; `.dev.vars` solo configura tu servidor local. El token compartido existente continúa funcionando.

## Datos y modelos

`scripts/export_nfl_web.py` une por `game_id` los archivos de `StrikeCast NFL/data/processed` y genera `public/data/nfl/*.json`. Ejecutar con Python que tenga pandas y pyarrow después de actualizar el pipeline NFL. El botón Actualizar datos NFL vuelve a leer esta exportación y consulta el scoreboard; no ejecuta modelos.

La exportación inicial conserva 3,300 partidos de 12 temporadas, filtros situacionales AND y la lógica original de ML, spread, totales y L/V/D. No modifica ni mejora el entrenamiento. El modelo principal llega hasta 2025; los partidos 2026 sin predicción se muestran como tales. El calendario/estado también puede estar desactualizado en los archivos originales.

Para añadir modelos de ganador: exportar sus probabilidades de victoria local por partido y registrar `{id,label,target:'winner',field}` en `models` del exportador. L/V/D es un modelo de margen, no un voto de ganador. Mercado se presenta sin margen de la casa, no como otro modelo.

## Telegram

Configurar `NFL_TELEGRAM_CHAT_ID="@StrikeCastNFL"` en `.dev.vars` local y, al desplegar, en las variables del servidor. Se reutiliza `TELEGRAM_BOT_TOKEN`; opcionalmente `NFL_TELEGRAM_BOT_TOKEN` lo sustituye. Reiniciar el servidor tras cambiar las variables si no se recargan automáticamente. Nunca se utiliza `TELEGRAM_CHAT_ID` como destino NFL.

El bot debe estar agregado al canal NFL con permiso para publicar. Cada pick abre una vista previa con número y momio editables, seguida de confirmación explícita. No se enviaron mensajes durante la verificación. Ante un error de conexión, revisar el canal antes de reintentar para evitar duplicados.

## Verificación

`node scripts/test_nfl_web.mjs`: validación de mensajes, mercados NFL, aislamiento de canales y consistencia de exportación. `tsc --noEmit` y `vinext build`: comprobaciones de compilación. No se ejecuta entrenamiento, despliegue ni envío real al canal.
