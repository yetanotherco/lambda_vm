#!/usr/bin/env node
//
// render_bench_comment.js — renders benchmark-pr.yml's PR-comment script offline.
//
// WHY: that comment is ~200 lines of JS embedded in YAML, and the only other way to
// see its output is to push and run a real benchmark on the shared bench server
// (~15-20 min of occupancy). This extracts the `script:` block from the workflow,
// runs it against synthetic env values, and prints the markdown — so a wording or
// formatting change is checked in a second.
//
// It renders; it does not assert. Read the output. The scenarios below are the ones
// that have actually broken before: sign handling on the verdict lines, the
// no-baseline and block-mismatch paths, and the fixture-unavailable footer.
//
// USAGE:  node scripts/render_bench_comment.js
//
// Add a scenario by adding a key to `scenarios`; each is the env the comment step
// would see. Keep one per branch of the renderer.
const fs = require('fs');
const path = require('path');
const wfPath = path.join(__dirname, '..', '.github', 'workflows', 'benchmark-pr.yml');
const wf = fs.readFileSync(wfPath, 'utf8');
const lines = wf.split('\n');
const start = lines.map(l => l.trim() === 'script: |').lastIndexOf(true);
if (start < 0) throw new Error('script block not found');
const indent = lines[start].match(/^\s*/)[0].length + 2;
const body = [];
for (let i = start + 1; i < lines.length; i++) {
  const l = lines[i];
  if (l.trim() === '') { body.push(''); continue; }
  if (l.match(/^\s*/)[0].length < indent) break;
  body.push(l.slice(indent));
}
const src = body.join('\n');

const captured = [];
const github = { rest: { issues: {
  listComments: async () => ({ data: [] }),
  createComment: async (a) => captured.push(a.body),
  updateComment: async (a) => captured.push(a.body),
}}};
const context = { repo: { owner: 'o', repo: 'r' }, issue: { number: 1 } };

const REAL = {
  COMMIT_SHA: 'c4e42b2900', REAL_EPOCH_LOG2: '22', BASELINE_SRC: 'cached',
  PR_REAL_TIME: '281.400', PR_REAL_PEAK: '32980', PR_REAL_EPOCHS: '13',
  PR_REAL_INPUT: 'ethrex_mainnet_25368371.bin',
  REAL_RUNS: '3', REAL_TIME_SPREAD: '1.9', REAL_ALL_TIMES: '279.1/281.4/284.4',
};
const scenarios = {
  'A: normal /bench — real block vs cached baseline': {
    ...REAL, BASE_REAL_TIME: '288.900', BASE_REAL_PEAK: '32975',
    REAL_TIME_DIFF: '-7.500', REAL_TIME_PCT: '-2.6', REAL_PEAK_DIFF: '5', REAL_PEAK_PCT: '0.0',
  },
  'B: clear improvement': {
    ...REAL, BASE_REAL_TIME: '340.000', BASE_REAL_PEAK: '32975',
    REAL_TIME_DIFF: '-58.600', REAL_TIME_PCT: '-17.2', REAL_PEAK_DIFF: '5', REAL_PEAK_PCT: '0.0',
  },
  'C: clear regression': {
    ...REAL, BASE_REAL_TIME: '240.000', BASE_REAL_PEAK: '32975',
    REAL_TIME_DIFF: '+41.400', REAL_TIME_PCT: '17.3', REAL_PEAK_DIFF: '5', REAL_PEAK_PCT: '0.0',
  },
  'D: unresolved middle band (3-10%)': {
    ...REAL, BASE_REAL_TIME: '300.000', BASE_REAL_PEAK: '32975',
    REAL_TIME_DIFF: '-18.600', REAL_TIME_PCT: '-6.2', REAL_PEAK_DIFF: '5', REAL_PEAK_PCT: '0.0',
  },
  'E: bad spread': {
    ...REAL, REAL_TIME_SPREAD: '11.4', REAL_ALL_TIMES: '265.0/281.4/297.1',
    BASE_REAL_TIME: '288.900', BASE_REAL_PEAK: '32975',
    REAL_TIME_DIFF: '-7.500', REAL_TIME_PCT: '-2.6', REAL_PEAK_DIFF: '5', REAL_PEAK_PCT: '0.0',
  },
  'F: no baseline yet': { ...REAL, BASELINE_SRC: 'built from main' },
  'G: baseline measured a different block': {
    ...REAL, BASE_REAL_TIME: '288.900', BASE_REAL_PEAK: '32975',
    REAL_MISMATCH: 'ethrex_hoodi_1265656.bin',
  },
  'H: fixture unavailable': { COMMIT_SHA: 'c4e42b2900', REAL_EPOCH_LOG2: '22', BASELINE_SRC: 'cached' },
  'I: with growth sweep (push-style)': {
    ...REAL, BASE_REAL_TIME: '288.900', BASE_REAL_PEAK: '32975',
    REAL_TIME_DIFF: '-7.500', REAL_TIME_PCT: '-2.6', REAL_PEAK_DIFF: '5', REAL_PEAK_PCT: '0.0',
    PR_GROWTH_HEAPS: '18756/26966/34748/43896/50431',
    PR_GROWTH_TIMES: '10.685/13.434/18.854/21.094/24.737',
    PR_GROWTH_SLOPE: '2007', PR_GROWTH_R2: '0.9981',
    BASE_GROWTH_HEAPS: '18700/26900/34700/43800/50217',
    BASE_GROWTH_SLOPE: '2000', BASE_GROWTH_R2: '0.9980',
    GROWTH_SLOPE_DIFF: '7', GROWTH_SLOPE_PCT: '0.4',
  },
};

(async () => {
  for (const [name, env] of Object.entries(scenarios)) {
    captured.length = 0;
    process.env = { ...env };
    const fn = new Function('github', 'context', 'require', `return (async () => { ${src} })()`);
    await fn(github, context, require);
    console.log(`\n${'='.repeat(72)}\n### ${name}\n${'='.repeat(72)}`);
    console.log(captured[0]);
  }
})();
