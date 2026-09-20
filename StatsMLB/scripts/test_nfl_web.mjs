import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
async function moduleFrom(path) {
  const { outputText } = ts.transpileModule(readFileSync(path, 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
  return import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
}
const { nflMessage, nflTelegramSettings } = await moduleFrom('lib/nfl-telegram.ts');
const pick = { gameId:'2026_02_NE_NYJ',home:'NYJ',away:'NE',kickoff:'2026-09-20T17:00:00Z',market:'ML',side:'NE',decimalOdds:1.91,pickNumber:1 };
assert.match(nflMessage(pick), /NE ML/);
assert.match(nflMessage({...pick,market:'OVER',side:'OVER',line:48.5}), /OVER 48.5/);
assert.match(nflMessage({...pick,market:'SPREAD',line:3.5}), /NE \+3.5/);
assert.match(nflMessage({...pick,market:'SPREAD',line:0}), /NE 0/);
assert.match(nflMessage({...pick,market:'LVD',side:'D'}), /6 puntos o menos/);
for (const invalid of [{decimalOdds:Infinity},{decimalOdds:0},{side:'DAL'},{home:'NYY'},{pickNumber:0},{kickoff:'bad'}]) assert.throws(()=>nflMessage({...pick,...invalid}));
assert.equal(nflTelegramSettings({TELEGRAM_BOT_TOKEN:'shared',TELEGRAM_CHAT_ID:'mlb'},{}).chat,undefined);
assert.deepEqual(nflTelegramSettings({TELEGRAM_BOT_TOKEN:'shared',TELEGRAM_CHAT_ID:'mlb'},{NFL_TELEGRAM_CHAT_ID:'@StrikeCastNFL'}),{token:'shared',chat:'@StrikeCastNFL'});
assert.equal(nflTelegramSettings({NFL_TELEGRAM_BOT_TOKEN:'nfl',TELEGRAM_BOT_TOKEN:'shared'},{}).token,'nfl');
const manifest=JSON.parse(readFileSync('public/data/nfl/index.json','utf8'));
let count=0;const ids=new Set();
for(const season of manifest.seasons){
  const {games}=JSON.parse(readFileSync(`public/data/nfl/${season.season}.json`,'utf8'));
  assert.equal(games.length,season.games);
  for(const g of games){assert(!ids.has(g.game_id));ids.add(g.game_id);assert(Array.isArray(g.filters));for(const field of ['model_home_win_prob','model_cover_home_prob','model_total_over_prob']) assert(g[field]===null || (g[field]>=0 && g[field]<=1));}
  count+=games.length;
}
assert.equal(count,3300);assert.equal(manifest.seasons.length,12);
assert(Object.keys(manifest.filters).length>0);
console.log('PASS: NFL payloads, validation, strict channel isolation, and 3,300 exported games / 12 seasons. No messages sent.');
const {normalizeNflLive,mergeNflLive,nflPeriodForDate}=await moduleFrom('lib/nfl-live.ts');
const raw={id:42,league_name:'nfl',season:2026,week:2,type:'reg',start_time:'2026-09-20T17:00:00Z',status:'complete',home_team_id:1,away_team_id:2,teams:[{id:1,abbr:'LAR'},{id:2,abbr:'JAC'}],boxscore:{total_home_points:28,total_away_points:24,period:4,clock:'00:00'},odds:[{type:'firsthalf',book_id:1,total:20},{type:'game',book_id:15,total:48.5,spread_home:-3.5,ml_home:-150,ml_away:130,inserted:'2026-09-20T12:00:00Z'}]};
const observed=normalizeNflLive({games:[raw]},'2026-09-20T21:00:00Z')[0];
assert.equal(observed.home_team,'LA');assert.equal(observed.away_team,'JAX');assert.equal(observed.home_score,28);assert.equal(observed.status,'final');assert.equal(observed.liveMarket.total,48.5);
const frozen={game_id:'2026_02_JAX_LA',season:2026,week:2,home_team:'LA',away_team:'JAX',kickoff_utc:raw.start_time,status:'upcoming',spread_line:7,total_line:52,model_home_win_prob:.6,model_cover_home_prob:.51,filters:['old_filter']};
assert.deepEqual(nflPeriodForDate([{...frozen,game_type:'REG'}],'2026-09-20',2026),{phase:'reg',week:2});
assert.equal(nflPeriodForDate([{...frozen,game_type:'REG'}],'2026-05-01',2026),null);
assert.deepEqual(nflPeriodForDate([{...frozen,game_type:'SB',week:22}],'2026-09-20',2026),{phase:'post',week:4});
assert.equal(normalizeNflLive({games:[{...raw,season:2025,type:'post',week:4}]},'now')[0].week,22);
const merged=mergeNflLive([frozen],[observed]);
assert.equal(merged.length,1);assert.equal(merged[0].home_score,28);assert.equal(merged[0].game_id,frozen.game_id);assert.equal(merged[0].spread_line,7);assert.equal(merged[0].total_line,52);assert.equal(merged[0].model_home_win_prob,.6);assert.equal(frozen.status,'upcoming');
assert.equal(mergeNflLive(merged,[{...observed,status:'upcoming',home_score:null}])[0].status,'final');
const added=mergeNflLive([],[observed])[0];assert.equal(added.model_home_win_prob,null);assert.deepEqual(added.filters,[]);
assert.equal(normalizeNflLive({games:[{...raw,status:'scheduled'}]},'now')[0].home_score,null);
assert.throws(()=>normalizeNflLive({oops:[]},'now'));assert.throws(()=>normalizeNflLive({games:[{...raw,league_name:'mlb'}]},'now'));
const noOdds=normalizeNflLive({games:[{...raw,odds:[{type:'game',book_id:15,ml_home:0,total:0}]}]},'now')[0];assert.equal(noOdds.liveMarket.home,null);assert.equal(noOdds.liveMarket.total,null);
console.log('PASS: live normalization, team aliases, no fabricated scores/picks, frozen model lines, and final-result regression protection.');
// Exercise the actual route with isolated storage/network dependencies; never
// access production data or Telegram in these contract tests.
const routeSource=readFileSync('app/api/nfl/live/route.ts','utf8').replace(/^import .*;\r?\n/gm,'');
const routeJs=ts.transpileModule(routeSource,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText.replace(/^export /gm,'');
const routeFactory=new Function('NextResponse','readNflLive','saveNflLive','normalizeNflLive','fetch',`${routeJs}\nreturn GET;`);
const jsonResponse={json:(data,init)=>new Response(JSON.stringify(data),init)};
const request=new Request('https://example.test/api/nfl/live?date=2026-09-20&season=2026&week=2');
const offline=async()=>{throw Error('offline');};
let handler=routeFactory(jsonResponse,async()=>({games:[observed],checkedAt:'2026-01-01T00:00:00Z'}),async()=>{},normalizeNflLive,offline);
let result=await (await handler(request)).json();assert.equal(result.state,'SAVED');assert.equal(result.games[0].home_score,28);assert(result.error);
handler=routeFactory(jsonResponse,offline,offline,normalizeNflLive,offline);
result=await (await handler(request)).json();assert.equal(result.state,'UNAVAILABLE');assert.equal(result.persistence,false);
handler=routeFactory(jsonResponse,async()=>({games:[observed],checkedAt:new Date().toISOString()}),offline,normalizeNflLive,async()=>{throw Error('cache should skip fetch');});
result=await (await handler(request)).json();assert.equal(result.state,'LIVE');assert.equal(result.error,undefined);
assert.equal((await handler(new Request('https://example.test/api/nfl/live?date=2026-02-30&season=2026&week=2'))).status,400);
let stored=[];
handler=routeFactory(jsonResponse,async()=>({games:[],checkedAt:null}),async(games)=>{stored=games;},normalizeNflLive,async()=>new Response(JSON.stringify({games:[raw]})));
result=await (await handler(request)).json();assert.equal(result.state,'LIVE');assert.equal(stored.length,1);assert.equal(result.persistence,true);
console.log('PASS: route cache, real-date validation, persistence, and saved/unavailable behavior on network failures.');
handler=routeFactory(jsonResponse,async()=>({games:[],checkedAt:null}),async()=>{},normalizeNflLive,async()=>new Response(JSON.stringify({games:[{...raw,week:1}]})));
result=await (await handler(request)).json();assert.equal(result.state,'UNAVAILABLE');assert(result.error);
console.log('PASS: explicit regular/postseason mapping and rejection of responses for the wrong week.');
