'use client';

import { Activity, ArrowUpRight, BarChart3, CalendarDays, CircleDot, FlaskConical, RefreshCw, ShieldCheck, Trophy, Wallet } from 'lucide-react';
import { Children, isValidElement, useEffect, useState, type KeyboardEvent, type ReactNode } from 'react';

const tabs = [
  { id: 'hoy', label: 'Jornada', icon: CalendarDays, title: 'Cada juego, en contexto.', description: 'Partidos, probabilidades y señales para preparar tus picks.', sections: ['hoy'] },
  { id: 'comparacion', label: 'Comparador', icon: BarChart3, title: 'Una vista. Todos los modelos.', description: 'Contrasta las selecciones, el mercado y el consenso por partido.', sections: ['comparacion'] },
  { id: 'rendimiento', label: 'Rendimiento', icon: Activity, title: 'La evidencia detrás del pick.', description: 'Explora resultados históricos y combinaciones con validación cronológica.', sections: ['rendimiento'] },
  { id: 'auditorias', label: 'Auditorías', icon: ShieldCheck, title: 'Pon cada señal a prueba.', description: 'Revisa muestras, resultados y límites de cada factor del modelo.', sections: ['auditoria-temporada-l10', 'auditoria-juego-anterior', 'auditoria-mejores-jugadores', 'auditoria-rotacion-coach', 'auditoria-calidad-rotacion', 'auditoria-fatiga-lineup', 'auditoria-cruces-lineup', 'auditoria-forma-rival'] },
  { id: 'equipos', label: 'Equipos y series', icon: Trophy, title: 'Lo que pasa después de una serie.', description: 'Rankings, cruces de contexto y respuesta de los equipos después de una barrida.', sections: ['rankings', 'escenarios', 'barridas-walkforward'] },
  { id: 'motor-momios', label: 'Momios', icon: Wallet, title: 'El mercado bajo la lupa.', description: 'Líneas de cierre, modelos de precio y diferencias frente al mercado.', sections: ['motor-momios'] },
  { id: 'metodo', label: 'Método', icon: FlaskConical, title: 'Entiende cómo se estima.', description: 'Metodología, procedencia de los datos y siguientes líneas de investigación.', sections: ['laboratorio'] },
];

const sectionLabels: Record<string, string> = {
  'auditoria-temporada-l10': 'Temporada y L10', 'auditoria-juego-anterior': 'Juego anterior',
  'auditoria-mejores-jugadores': 'Mejores jugadores', 'auditoria-rotacion-coach': 'Rotación del coach',
  'auditoria-calidad-rotacion': 'Calidad del lineup', 'auditoria-fatiga-lineup': 'Fatiga',
  'auditoria-cruces-lineup': 'Cruces de señales', 'auditoria-forma-rival': 'Forma vs. rival',
  rankings: 'Rankings', escenarios: 'Escenarios', 'barridas-walkforward': 'Barridas walk-forward',
};

function resolveRoute(hash: string) {
  const id = hash.replace(/^#/, '');
  const tab = tabs.find(item => item.id === id || item.sections.includes(id)) ?? tabs[0];
  return { tab: tab.id, section: tab.sections.includes(id) ? id : tab.sections[0] };
}

export function moveFocus(event: KeyboardEvent<HTMLDivElement>) {
  if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
  const buttons = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
  const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
  if (index < 0) return;
  event.preventDefault();
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
  buttons[next].focus();
  buttons[next].click();
}

type Props = {
  children: ReactNode;
  metrics: { label: string; value: string; detail: string }[];
  cutoff: string;
  combination: string;
  telegramConfigured: boolean;
  updating: boolean;
  onRefresh: () => void;
  source: ReactNode;
  feedback: ReactNode;
};

export default function DashboardTabs({ children, metrics, cutoff, combination, telegramConfigured, updating, onRefresh, source, feedback }: Props) {
  const [route, setRoute] = useState(() => resolveRoute(''));
  const [subsections, setSubsections] = useState<Record<string, string>>({});
  useEffect(() => {
    const sync = () => {
      if (['#dashboard-content', '#inicio'].includes(window.location.hash)) return;
      const next = resolveRoute(window.location.hash);
      setRoute(next);
      setSubsections(current => ({ ...current, [next.tab]: next.section }));
    };
    sync();
    window.addEventListener('hashchange', sync);
    window.addEventListener('popstate', sync);
    return () => { window.removeEventListener('hashchange', sync); window.removeEventListener('popstate', sync); };
  }, []);

  const selected = tabs.find(tab => tab.id === route.tab) ?? tabs[0];
  const navigate = (tab: typeof tabs[number], section = subsections[tab.id] ?? tab.sections[0]) => {
    setRoute({ tab: tab.id, section });
    setSubsections(current => ({ ...current, [tab.id]: section }));
    // Hash navigation supports bookmarked views and the browser's Back button.
    if (window.location.hash !== `#${section}`) window.history.pushState(null, '', `#${section}`);
  };
  const content = Children.toArray(children).filter(isValidElement<{ id?: string; 'data-section'?: string }>);

  return <div className="dashboard app-content">
    <a className="skip-link" href="#dashboard-content">Saltar al contenido</a>
    <header className="dashboard-header" id="inicio">
      <a className="dashboard-brand" href="#hoy" aria-label="StatsMLB, ir a Jornada"><span className="dashboard-brand-icon"><CircleDot size={24} /></span><span>STRIKE<span>CAST</span><small>STATSMLB / CENTRO DE ANÁLISIS</small></span></a>
      <div className="dashboard-header-right"><span className={`connection-status ${telegramConfigured ? 'connected' : ''}`}><i />{telegramConfigured ? 'Telegram conectado' : 'Telegram pendiente'}</span><button className="dashboard-refresh" onClick={onRefresh} disabled={updating}><RefreshCw size={15} className={updating ? 'spin' : ''} />{updating ? 'Actualizando…' : 'Actualizar datos'}</button></div>
    </header>
    <main className="dashboard-body" id="dashboard-content" tabIndex={-1}>
      <div className="dashboard-intro"><div><span className="dashboard-overline">MLB INTELLIGENCE <span>/</span> SERIE & CONTEXTO</span><h1>{selected.title}</h1><p>{selected.description}</p></div><div className="dashboard-source">{source}<small>Histórico cerrado al {cutoff}</small></div></div>
      <div className="dashboard-metrics">{metrics.map((metric, index) => <article key={metric.label}><div><span>{metric.label}</span><ArrowUpRight size={15} /></div><strong>{metric.value}</strong><small>{metric.detail}</small><i className={`metric-accent metric-accent-${index}`} /></article>)}</div>
      <div className="dashboard-tabs" role="tablist" aria-label="Secciones de StatsMLB" onKeyDown={moveFocus}>{tabs.map(tab => <button type="button" role="tab" key={tab.id} id={`tab-${tab.id}`} aria-selected={selected.id === tab.id} aria-controls={`panel-${tab.id}`} tabIndex={selected.id === tab.id ? 0 : -1} onClick={() => navigate(tab)}><tab.icon size={17} /><span>{tab.label}</span></button>)}</div>
      <div className="dashboard-model"><span><CircleDot size={14} /> Modelo activo</span><strong>{combination || 'Sin factores activos'}</strong><a href="#rendimiento">Configurar <ArrowUpRight size={13} /></a></div>
      <div aria-live="polite">{feedback}</div>
      {tabs.map(tab => {
        const section = selected.id === tab.id ? route.section : subsections[tab.id] ?? tab.sections[0];
        return <div key={tab.id} id={`panel-${tab.id}`} className="dashboard-panel" role="tabpanel" aria-labelledby={`tab-${tab.id}`} hidden={selected.id !== tab.id} tabIndex={0}>
          {tab.sections.length > 1 && <div className="dashboard-subtabs" role="tablist" aria-label={`Vistas de ${tab.label}`} onKeyDown={moveFocus}>{tab.sections.map(id => <button type="button" key={id} role="tab" id={`subtab-${id}`} aria-controls={`subpanel-${id}`} aria-selected={section === id} tabIndex={section === id ? 0 : -1} onClick={() => navigate(tab, id)}>{sectionLabels[id]}</button>)}</div>}
          {tab.sections.map(id => {
            const nodes = content.filter(child => (child.props['data-section'] ?? child.props.id) === id);
            return <div key={id} id={`subpanel-${id}`} hidden={tab.sections.length > 1 && section !== id} {...(tab.sections.length > 1 ? { role: 'tabpanel', 'aria-labelledby': `subtab-${id}`, tabIndex: 0 } : {})}>
              {nodes.length ? nodes : <div className="empty-state"><FlaskConical size={28} /><h3>Datos no disponibles</h3><p>Este análisis no tiene datos cargados. Puedes volver a consultar con Actualizar datos.</p></div>}
            </div>;
          })}
        </div>;
      })}
      <footer className="dashboard-footer"><span><CircleDot size={14} /> StatsMLB <b>Datos antes que intuición.</b></span><p>STRIKECAST · MLB StatsAPI · Estimaciones informativas, no garantías.</p><a href="#inicio" onClick={event => { event.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); }}>Volver arriba ↑</a></footer>
    </main>
  </div>;
}
