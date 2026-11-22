'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const CASES = path.join(__dirname, 'parity_cases.json');

test('JavaScript scores match the Python model', { skip: !fs.existsSync(CASES) && 'run src/export_model.py first' }, () => {
  const detector = require('../security/detector');
  const cases = JSON.parse(fs.readFileSync(CASES, 'utf8'));
  const maxDiff = Math.max(...cases.map((c) => Math.abs(detector.score(c.text) - c.py)));
  assert.ok(maxDiff < 1e-9, `maximum difference ${maxDiff}`);
});
