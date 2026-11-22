'use strict';

const MAX_ENTRIES = 200;

class IncidentLog {
  constructor() {
    this.scored = 0;
    this.blockedTotal = 0;
    this.blocked = [];
  }

  countScored() {
    this.scored += 1;
  }

  add(entry) {
    this.blockedTotal += 1;
    this.blocked.unshift({ time: new Date(), ...entry });
    if (this.blocked.length > MAX_ENTRIES) this.blocked.pop();
  }
}

module.exports = IncidentLog;
