'use strict';

const express = require('express');
const requireAuth = require('../middleware/requireAuth');

function createRouter({ home, auth, consolePage, incidents, protectionEnabled }) {
  const router = express.Router();
  const loginGuards = protectionEnabled
    ? [require('../middleware/sqliGuard')(['username', 'password'], incidents)]
    : [];

  router.get('/', home.index);
  router.post('/check', home.check);
  router.get('/login', auth.showLogin);
  router.post('/login', ...loginGuards, auth.login);
  router.post('/logout', auth.logout);
  router.get('/console', requireAuth, consolePage.index);

  return router;
}

module.exports = createRouter;
