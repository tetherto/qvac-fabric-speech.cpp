(function (root) {
  const DATA_DIR = 'data/';
  const RESIZE_DEBOUNCE_MS = 150;
  const ROW_KEYS = ['rows', 'data', 'results'];
  const STATUS_OK = 'ok';
  const STATUS_LOADING = 'loading';
  const STATUS_MISSING = 'missing';
  const STATUS_ERROR = 'error';

  function rowsOf(value) {
    if (Array.isArray(value)) return value;
    const key = value && typeof value === 'object' ? ROW_KEYS.find((k) => Array.isArray(value[k])) : undefined;
    return key ? value[key] : value;
  }

  function columnsOf(data) {
    return Array.isArray(data) && data.length && data[0] && typeof data[0] === 'object' ? Object.keys(data[0]) : [];
  }

  function mergeParams(current, next) {
    const merged = { ...current };
    Object.entries(next || {}).forEach(([key, value]) => {
      if (value === null || value === undefined) delete merged[key];
      else merged[key] = value;
    });
    return merged;
  }

  function createDash(env) {
    const sources = new Map();
    const calcs = new Map();
    const drawings = new Map();
    const listeners = [];
    let params = {};
    let version = 0;
    let queued = false;

    function runListeners() {
      listeners.forEach((fn) => {
        try { fn(); } catch (error) { env.report('dash.onData', error); }
      });
    }

    function drawName(name) {
      const fn = drawings.get(name);
      if (!fn) return;
      env.elements(name).forEach((el) => {
        try { fn(el); } catch (error) { env.report('dash.draw ' + name, error); }
      });
    }

    function drawAll() {
      drawings.forEach((fn, name) => drawName(name));
    }

    function render() {
      queued = false;
      runListeners();
      drawAll();
    }

    function schedule() {
      if (queued) return;
      queued = true;
      env.defer(render);
    }

    function settle(entry, status, data, message) {
      Object.assign(entry, { status, data, message });
      version += 1;
      schedule();
    }

    function load(id) {
      const entry = { status: STATUS_LOADING, data: [], message: '' };
      sources.set(id, entry);
      env.fetchJson(DATA_DIR + encodeURIComponent(id) + '.json').then(
        (value) => settle(entry, STATUS_OK, rowsOf(value), ''),
        (error) => settle(entry, error && error.missing ? STATUS_MISSING : STATUS_ERROR, [], String((error && error.message) || error)));
      return entry;
    }

    function inputValue(id, state) {
      if (calcs.has(id)) return state.data;
      return Array.isArray(state.data) ? state.data : [];
    }

    function calcResult(calc, states) {
      if (calc.stamp === version) return calc.result;
      try {
        calc.result = { status: STATUS_OK, data: calc.fn(...calc.inputs.map((id, i) => inputValue(id, states[i]))), message: '' };
      } catch (error) {
        calc.result = { status: STATUS_ERROR, data: null, message: String(error) };
      }
      calc.stamp = version;
      return calc.result;
    }

    function calcState(id) {
      const calc = calcs.get(id);
      const states = calc.inputs.map((input) => read(input));
      const blocked = states.find((state) => state.status !== STATUS_OK);
      return blocked ? { status: blocked.status, data: null, message: blocked.message } : calcResult(calc, states);
    }

    function read(id) {
      if (calcs.has(id)) return calcState(id);
      return sources.get(id) || load(id);
    }

    function snapshot(state) {
      return { status: state.status, data: state.data, message: state.message || '', refreshing: false, meta: {}, columns: columnsOf(state.data) };
    }

    const api = {
      data(id) { return snapshot(read(id)); },
      onData(fn) {
        listeners.push(fn);
        try { fn(); } catch (error) { env.report('dash.onData', error); }
      },
      calc(id, spec) {
        calcs.set(id, { inputs: spec.inputs || [], fn: spec.fn, stamp: -1, result: null });
        schedule();
      },
      draw(name, fn) {
        if (typeof fn !== 'function') { drawName(name); return; }
        drawings.set(name, fn);
        schedule();
      },
      params() { return { ...params }; },
      setParams(next) {
        params = mergeParams(params, next);
        schedule();
      },
      setLink(name) { env.setHash(name); },
    };
    return { api, rerender: schedule };
  }

  function fetchJson(url) {
    return fetch(url, { cache: 'no-cache' }).then((response) => {
      if (response.status === 404) throw Object.assign(new Error('no data file ' + url), { missing: true });
      if (!response.ok) throw new Error('HTTP ' + response.status + ' for ' + url);
      return response.json();
    });
  }

  function browserEnv() {
    return {
      fetchJson,
      elements: (name) => Array.from(document.querySelectorAll('[data-draw="' + CSS.escape(name) + '"]')),
      defer: (fn) => setTimeout(fn, 0),
      setHash: (name) => history.replaceState(null, '', '#' + encodeURIComponent(name)),
      report: (where, error) => console.error(where, error),
    };
  }

  function bindResize(rerender) {
    let timer = 0;
    let width = document.documentElement.clientWidth;
    window.addEventListener('resize', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const next = document.documentElement.clientWidth;
        if (next === width) return;
        width = next;
        rerender();
      }, RESIZE_DEBOUNCE_MS);
    });
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = { createDash, rowsOf, mergeParams };
    return;
  }
  const dash = createDash(browserEnv());
  root.dash = dash.api;
  bindResize(dash.rerender);
})(typeof window === 'object' ? window : globalThis);
