import { index, integer, sqliteTable, text, uniqueIndex } from 'drizzle-orm/sqlite-core';

export const dailyGames = sqliteTable('daily_games', {
  gamePk: integer('game_pk').primaryKey(),
  officialDate: text('official_date').notNull(),
  gameDate: text('game_date').notNull(),
  status: text('status').notNull(),
  abstractState: text('abstract_state').notNull(),
  venue: text('venue'),
  awayTeam: text('away_team').notNull(),
  awayName: text('away_name').notNull(),
  awayScore: integer('away_score'),
  awayPitcher: text('away_pitcher'),
  awayLineupJson: text('away_lineup_json'),
  homeTeam: text('home_team').notNull(),
  homeName: text('home_name').notNull(),
  homeScore: integer('home_score'),
  homePitcher: text('home_pitcher'),
  homeLineupJson: text('home_lineup_json'),
  source: text('source').notNull(),
  rawJson: text('raw_json').notNull(),
  firstSeenAt: text('first_seen_at').notNull(),
  updatedAt: text('updated_at').notNull(),
  finalAt: text('final_at'),
}, table => [
  index('idx_daily_games_official_date').on(table.officialDate, table.gameDate),
  index('idx_daily_games_final_at').on(table.finalAt),
]);

export const oddsSnapshots = sqliteTable('odds_snapshots', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  gamePk: integer('game_pk'),
  officialDate: text('official_date').notNull(),
  sourceEventId: text('source_event_id').notNull(),
  commenceTime: text('commence_time').notNull(),
  awayTeam: text('away_team').notNull(),
  homeTeam: text('home_team').notNull(),
  sportsbook: text('sportsbook').notNull(),
  awayAmerican: integer('away_american').notNull(),
  homeAmerican: integer('home_american').notNull(),
  sourceUpdatedAt: text('source_updated_at').notNull(),
  capturedAt: text('captured_at').notNull(),
  source: text('source').notNull(),
  rawJson: text('raw_json').notNull(),
}, table => [
  uniqueIndex('uq_odds_snapshot_observation').on(
    table.sourceEventId,
    table.sportsbook,
    table.sourceUpdatedAt,
    table.awayAmerican,
    table.homeAmerican,
  ),
  index('idx_odds_snapshots_date_game').on(table.officialDate, table.gamePk, table.capturedAt),
]);

export const refreshRuns = sqliteTable('refresh_runs', {
  id: text('id').primaryKey(),
  startedAt: text('started_at').notNull(),
  finishedAt: text('finished_at'),
  status: text('status').notNull(),
  todayDate: text('today_date').notNull(),
  tomorrowDate: text('tomorrow_date').notNull(),
  gamesSeen: integer('games_seen').notNull().default(0),
  finalGames: integer('final_games').notNull().default(0),
  oddsEvents: integer('odds_events').notNull().default(0),
  oddsBooks: integer('odds_books').notNull().default(0),
  errorsJson: text('errors_json').notNull().default('[]'),
}, table => [index('idx_refresh_runs_started_at').on(table.startedAt)]);
