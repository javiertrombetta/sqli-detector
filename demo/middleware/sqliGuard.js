'use strict';

const crypto = require('crypto');
const detector = require('../security/detector');
const project = require('../config/project');
const { formText } = require('../helpers/text');

const MAX_LOGGED_VALUE = 80;

function sqliGuard(fields, incidents) {
  return (req, res, next) => {
    const checks = fields.map((field) => ({ field, ...detector.inspect(formText(req.body[field])) }));
    incidents.countScored();
    req.detectorCheck = {
      threshold: detector.info.threshold,
      fields: checks.map(({ field, score }) => ({ field, score })),
    };
    const flagged = checks.find((check) => check.attack);
    if (!flagged) return next();

    const isPassword = flagged.field === 'password';
    const value = isPassword ? null : formText(req.body[flagged.field]).slice(0, MAX_LOGGED_VALUE);
    const reference = `INC-${crypto.randomBytes(4).toString('hex').toUpperCase()}`;
    incidents.add({ reference, field: flagged.field, score: flagged.score, value, source: req.ip });
    console.warn(`[blocked] ${reference} field=${flagged.field} score=${flagged.score.toFixed(4)} ` +
                 `threshold=${flagged.threshold.toFixed(4)} value=${isPassword ? '(not logged)' : JSON.stringify(value)} ip=${req.ip}`);
    return res.status(403).render('auth/blocked', {
      title: `Request blocked | ${project.siteName}`,
      reference,
      check: { ...flagged, evidence: isPassword ? [] : flagged.evidence },
    });
  };
}

module.exports = sqliGuard;
