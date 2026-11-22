'use strict';

const fs = require('fs');
const path = require('path');

const MODEL_PATH = path.join(__dirname, 'model.json');
if (!fs.existsSync(MODEL_PATH)) {
  throw new Error('security/model.json not found. Run "python export_model.py" from src/ first.');
}
const model = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf8'));

function charWbNgrams(text) {
  const grams = [];
  const words = String(text).split(/\s+/).filter((w) => w.length > 0);
  for (const word of words) {
    const w = ` ${word} `;
    for (let n = model.ngram_min; n <= model.ngram_max; n++) {
      let offset = 0;
      grams.push(w.slice(offset, offset + n));
      while (offset + n < w.length) {
        offset += 1;
        grams.push(w.slice(offset, offset + n));
      }
      if (offset === 0) break;
    }
  }
  return grams;
}

function tfidf(text) {
  const counts = {};
  for (const gram of charWbNgrams(text)) {
    const idx = model.vocabulary[gram];
    if (idx !== undefined) counts[idx] = (counts[idx] || 0) + 1;
  }
  let norm = 0;
  const values = {};
  for (const idx of Object.keys(counts)) {
    values[idx] = counts[idx] * model.idf[idx];
    norm += values[idx] ** 2;
  }
  norm = Math.sqrt(norm) || 1;
  for (const idx of Object.keys(values)) values[idx] /= norm;
  return values;
}

function score(text) {
  const values = tfidf(text);
  let z = model.intercept;
  for (const idx of Object.keys(values)) z += values[idx] * model.coef[idx];
  return 1 / (1 + Math.exp(-z));
}

let ngramOf = null;

function inspect(text, limit = 5) {
  if (!ngramOf) {
    ngramOf = new Array(model.idf.length);
    for (const [gram, idx] of Object.entries(model.vocabulary)) ngramOf[idx] = gram;
  }
  const values = tfidf(text);
  let z = model.intercept;
  const contributions = [];
  for (const idx of Object.keys(values)) {
    const contribution = values[idx] * model.coef[idx];
    z += contribution;
    contributions.push({ ngram: ngramOf[idx], contribution });
  }
  const probability = 1 / (1 + Math.exp(-z));
  return {
    score: probability,
    threshold: model.threshold,
    attack: probability >= model.threshold,
    evidence: contributions
      .filter((item) => item.contribution > 0)
      .sort((x, y) => y.contribution - x.contribution)
      .slice(0, limit),
  };
}

module.exports = {
  score,
  inspect,
  info: {
    threshold: model.threshold,
    vocabularySize: model.idf.length,
    ngramRange: [model.ngram_min, model.ngram_max],
  },
};
