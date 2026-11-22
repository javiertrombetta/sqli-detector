'use strict';

const crypto = require('crypto');

const argv = process.argv.slice(2);

module.exports = {
  port: Number(process.env.PORT) || 3000,
  protectionEnabled: !argv.includes('--vulnerable') && process.env.LOGIN_PROTECTION !== 'off',
  session: {
    name: 'sqlid.sid',
    secret: process.env.SESSION_SECRET || crypto.randomBytes(32).toString('hex'),
    rememberMs: 7 * 24 * 60 * 60 * 1000,
  },
  timeZone: 'Pacific/Auckland',
};
