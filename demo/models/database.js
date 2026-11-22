'use strict';

const crypto = require('crypto');
const Database = require('better-sqlite3');

function passwordDigest(password) {
  return crypto.createHash('sha256').update(String(password)).digest('hex');
}

function createDatabase() {
  const db = new Database(':memory:');

  db.exec(`
    CREATE TABLE users (
      id            INTEGER PRIMARY KEY,
      username      TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      full_name     TEXT NOT NULL
    );
  `);

  const addUser = db.prepare(`INSERT INTO users (username, password_hash, full_name)
                              VALUES (?, ?, ?)`);
  addUser.run('researcher', passwordDigest('researcher123'), 'Researcher');
  addUser.run('analyst', passwordDigest('analyst123'), 'Security Analyst');
  addUser.run('operator', passwordDigest("' OR '1'='1"), 'Operator');

  return db;
}

module.exports = { createDatabase, passwordDigest };
