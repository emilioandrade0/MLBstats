import Link from 'next/link';
export default function SportSwitch({ sport }: { sport: 'MLB' | 'NFL' }) {
  return <nav aria-label="Deporte" className="sport-switch"><Link href="/#hoy" aria-current={sport === 'MLB' ? 'page' : undefined}>⚾ MLB</Link><Link href="/nfl" aria-current={sport === 'NFL' ? 'page' : undefined}>🏈 NFL</Link></nav>;
}
