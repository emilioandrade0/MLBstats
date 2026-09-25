/** Follow the comparison rows by game ID (including separate doubleheader games).
 * Keep unmatched market games at the end; never discard their observations.
 */
export function orderLineMovement<T extends { commence: string }>(
  games: Record<string, T>, comparisonIds: readonly (string | number)[],
): [string, T][] {
  const position = new Map(comparisonIds.map((id, index) => [String(id), index]));
  const time = (value: string) => {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
  };
  return Object.entries(games).sort(([aId, a], [bId, b]) =>
    (position.get(aId) ?? Number.MAX_SAFE_INTEGER) - (position.get(bId) ?? Number.MAX_SAFE_INTEGER)
    || time(a.commence) - time(b.commence)
    || aId.localeCompare(bId, 'en', { numeric: true }),
  );
}
