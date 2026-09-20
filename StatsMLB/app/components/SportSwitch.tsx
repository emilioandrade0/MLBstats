export default function SportSwitch({ sport }: { sport: 'MLB' | 'NFL' }) {
  return <nav aria-label="Deporte" className="sport-switch"><a href="/#hoy" aria-current={sport === 'MLB' ? 'page' : undefined}>⚾ MLB</a><a href="/nfl" aria-current={sport === 'NFL' ? 'page' : undefined}>🏈 NFL</a></nav>;
}
