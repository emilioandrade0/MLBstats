import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'StatsMLB · Serie & contexto',
  description: 'Dashboard explicable de juegos MLB, barridas, respuesta posterior y probabilidad estimada de victoria.',
  openGraph: {
    title: 'StatsMLB · Serie & contexto',
    description: 'Juegos de hoy, barridas y probabilidad explicable de victoria.',
    images: [{ url: '/og-statsmlb.png', width: 1200, height: 630, alt: 'StatsMLB, serie y contexto' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'StatsMLB · Serie & contexto',
    description: 'Juegos de hoy, barridas y probabilidad explicable de victoria.',
    images: ['/og-statsmlb.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="es"><body>{children}</body></html>;
}
