'use strict';

const { timeZone } = require('../config');

function clockTime(date) {
  return new Intl.DateTimeFormat('en-NZ',
    { timeZone, hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }).format(date);
}

module.exports = { clockTime };
