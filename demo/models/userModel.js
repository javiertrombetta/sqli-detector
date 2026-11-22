'use strict';

const { passwordDigest } = require('./database');

class UserModel {
  constructor(db) {
    this.db = db;
    this.byCredentials = db.prepare(
      'SELECT id, username, full_name FROM users WHERE username = ? AND password_hash = ?');
  }

  findByCredentials(username, password) {
    return this.byCredentials.get(username, passwordDigest(password)) || null;
  }

  findByCredentialsUnsafe(username, password) {
    const sql = `SELECT id, username, full_name FROM users
                 WHERE username = '${username}' AND password_hash = '${passwordDigest(password)}'`;
    console.log(`[query] ${sql.replace(/\s+/g, ' ')}`);
    return this.db.prepare(sql).get() || null;
  }
}

module.exports = UserModel;
