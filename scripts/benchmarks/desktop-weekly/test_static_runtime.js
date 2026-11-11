const assert = require('node:assert/strict');
const path = require('node:path');
const { createDash, rowsOf, mergeParams } = require(path.join(__dirname, 'dashboard', 'static-runtime.js'));

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

function fakeEnv(files, failures = {}) {
  const queue = [];
  const elementsByName = new Map();
  const env = {
    requests: [],
    hashes: [],
    reports: [],
    fetchJson(url) {
      env.requests.push(url);
      if (url in failures) return Promise.reject(new Error(failures[url]));
      if (url in files) return Promise.resolve(files[url]);
      return Promise.reject(Object.assign(new Error('no data file ' + url), { missing: true }));
    },
    elements: (name) => elementsByName.get(name) || [],
    defer: (fn) => queue.push(fn),
    setHash: (name) => env.hashes.push(name),
    report: (where) => env.reports.push(where),
    place: (name, elements) => elementsByName.set(name, elements),
    flush: () => queue.splice(0).forEach((fn) => fn()),
  };
  return env;
}

async function settle(env) {
  await new Promise((resolve) => setImmediate(resolve));
  env.flush();
}

test('a dataset loads from its data file with its rows and columns', async () => {
  const env = fakeEnv({ 'data/runs.json': [{ id: 'a', speed_x: 2 }, { id: 'b', speed_x: 3 }], 'data/machine.json': { rows: [{ machine: 'RTX 5090' }] } });
  const { api } = createDash(env);
  assert.equal(api.data('runs').status, 'loading');
  assert.deepEqual(api.data('runs').data, []);
  await settle(env);
  assert.equal(api.data('runs').status, 'ok');
  assert.deepEqual(api.data('runs').columns, ['id', 'speed_x']);
  api.data('machine');
  await settle(env);
  assert.deepEqual(api.data('machine').data, [{ machine: 'RTX 5090' }]);
  assert.deepEqual(env.requests, ['data/runs.json', 'data/machine.json']);
});

test('a missing data file and a failed fetch each give their own status', async () => {
  const env = fakeEnv({}, { 'data/history.json': 'HTTP 500 for data/history.json' });
  const { api } = createDash(env);
  api.data('coverage');
  api.data('history');
  await settle(env);
  assert.equal(api.data('coverage').status, 'missing');
  assert.equal(api.data('history').status, 'error');
  assert.equal(api.data('history').message, 'HTTP 500 for data/history.json');
  assert.equal(api.data('unknown').status, 'loading');
});

test('a calc waits for its inputs, works once per data change and takes another calc as its value', async () => {
  const env = fakeEnv({ 'data/runs.json': [{ id: 'a' }, { id: 'b' }] });
  const { api } = createDash(env);
  let calls = 0;
  api.calc('total', { inputs: ['runs'], fn: (runs) => { calls += 1; return runs.length; } });
  api.calc('double', { inputs: ['total'], fn: (total) => total * 2 });
  assert.equal(api.data('total').status, 'loading');
  assert.equal(api.data('total').data, null);
  await settle(env);
  assert.equal(api.data('total').data, 2);
  assert.equal(api.data('double').data, 4);
  api.data('total');
  assert.equal(calls, 1);
});

test('a calc over a source that is not rows gets an empty list', async () => {
  const env = fakeEnv({ 'data/meta.json': { title: 'not rows' } });
  const { api } = createDash(env);
  api.calc('count', { inputs: ['meta'], fn: (meta) => meta.length });
  api.data('count');
  await settle(env);
  assert.deepEqual(api.data('count'), { status: 'ok', data: 0, message: '', refreshing: false, meta: {}, columns: [] });
});

test('drawings run for every element on render and on demand, and one that throws stops no other', () => {
  const env = fakeEnv({});
  const { api } = createDash(env);
  const first = { n: 0 };
  const second = { n: 0 };
  env.place('fact', [first, second]);
  env.place('broken', [{}]);
  api.draw('broken', () => { throw new Error('boom'); });
  api.draw('fact', (el) => { el.n += 1; });
  env.flush();
  assert.deepEqual([first.n, second.n], [1, 1]);
  assert.deepEqual(env.reports, ['dash.draw broken']);
  api.draw('fact');
  assert.equal(first.n, 2);
});

test('onData runs at once and once per render, and filters merge with null removing a key', () => {
  const env = fakeEnv({});
  const { api, rerender } = createDash(env);
  let runs = 0;
  api.onData(() => { runs += 1; });
  assert.equal(runs, 1);
  api.setParams({ tab: 'asr', model: 'x' });
  api.setParams({ model: null, metric: 'rtf' });
  env.flush();
  assert.equal(runs, 2);
  assert.deepEqual(api.params(), { tab: 'asr', metric: 'rtf' });
  const copy = api.params();
  copy.tab = 'changed';
  assert.equal(api.params().tab, 'asr');
  rerender();
  env.flush();
  assert.equal(runs, 3);
});

test('setLink names the view in the address', () => {
  const env = fakeEnv({});
  createDash(env).api.setLink('runs');
  assert.deepEqual(env.hashes, ['runs']);
});

test('rows are found bare or under rows, data or results, and params merge without changing the old ones', () => {
  assert.deepEqual(rowsOf([1]), [1]);
  assert.deepEqual(rowsOf({ data: [2] }), [2]);
  assert.deepEqual(rowsOf({ results: [3] }), [3]);
  assert.deepEqual(rowsOf({ other: 4 }), { other: 4 });
  const before = { a: 1 };
  assert.deepEqual(mergeParams(before, { a: undefined, b: 2 }), { b: 2 });
  assert.deepEqual(before, { a: 1 });
});

async function runTests() {
  let failed = 0;
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log('ok - ' + name);
    } catch (error) {
      failed += 1;
      console.log('not ok - ' + name + '\n' + error.stack);
    }
  }
  process.exitCode = failed ? 1 : 0;
}

runTests();
