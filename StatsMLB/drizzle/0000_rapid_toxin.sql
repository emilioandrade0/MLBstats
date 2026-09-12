CREATE TABLE `daily_games` (
	`game_pk` integer PRIMARY KEY NOT NULL,
	`official_date` text NOT NULL,
	`game_date` text NOT NULL,
	`status` text NOT NULL,
	`abstract_state` text NOT NULL,
	`venue` text,
	`away_team` text NOT NULL,
	`away_name` text NOT NULL,
	`away_score` integer,
	`away_pitcher` text,
	`away_lineup_json` text,
	`home_team` text NOT NULL,
	`home_name` text NOT NULL,
	`home_score` integer,
	`home_pitcher` text,
	`home_lineup_json` text,
	`source` text NOT NULL,
	`raw_json` text NOT NULL,
	`first_seen_at` text NOT NULL,
	`updated_at` text NOT NULL,
	`final_at` text
);
--> statement-breakpoint
CREATE INDEX `idx_daily_games_official_date` ON `daily_games` (`official_date`,`game_date`);--> statement-breakpoint
CREATE INDEX `idx_daily_games_final_at` ON `daily_games` (`final_at`);--> statement-breakpoint
CREATE TABLE `odds_snapshots` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`game_pk` integer,
	`official_date` text NOT NULL,
	`source_event_id` text NOT NULL,
	`commence_time` text NOT NULL,
	`away_team` text NOT NULL,
	`home_team` text NOT NULL,
	`sportsbook` text NOT NULL,
	`away_american` integer NOT NULL,
	`home_american` integer NOT NULL,
	`source_updated_at` text NOT NULL,
	`captured_at` text NOT NULL,
	`source` text NOT NULL,
	`raw_json` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `uq_odds_snapshot_observation` ON `odds_snapshots` (`source_event_id`,`sportsbook`,`source_updated_at`,`away_american`,`home_american`);--> statement-breakpoint
CREATE INDEX `idx_odds_snapshots_date_game` ON `odds_snapshots` (`official_date`,`game_pk`,`captured_at`);--> statement-breakpoint
CREATE TABLE `refresh_runs` (
	`id` text PRIMARY KEY NOT NULL,
	`started_at` text NOT NULL,
	`finished_at` text,
	`status` text NOT NULL,
	`today_date` text NOT NULL,
	`tomorrow_date` text NOT NULL,
	`games_seen` integer DEFAULT 0 NOT NULL,
	`final_games` integer DEFAULT 0 NOT NULL,
	`odds_events` integer DEFAULT 0 NOT NULL,
	`odds_books` integer DEFAULT 0 NOT NULL,
	`errors_json` text DEFAULT '[]' NOT NULL
);
--> statement-breakpoint
CREATE INDEX `idx_refresh_runs_started_at` ON `refresh_runs` (`started_at`);