'use strict';

const config = require('../config');
const project = require('../config/project');
const { formText } = require('../helpers/text');

function createAuthController({ users, protectionEnabled }) {
  const findUser = protectionEnabled
    ? (username, password) => users.findByCredentials(username, password)
    : (username, password) => users.findByCredentialsUnsafe(username, password);

  const renderLogin = (res, status, error = null, username = '') =>
    res.status(status).render('auth/login', { title: `Sign in | ${project.siteName}`, error, username });

  return {
    showLogin(req, res) {
      if (req.session.user) return res.redirect('/console');
      return renderLogin(res, 200);
    },

    login(req, res, next) {
      const username = formText(req.body.username);
      const password = formText(req.body.password);

      let user = null;
      try {
        user = findUser(username, password);
      } catch (err) {
        console.warn(`[sql error] ${err.message}`);
      }
      if (!user) return renderLogin(res, 401, 'The username or password is incorrect.', username);

      return req.session.regenerate((err) => {
        if (err) return next(err);
        req.session.user = {
          id: user.id,
          username: String(user.username || username),
          fullName: String(user.full_name || user.username || username),
        };
        req.session.detectorCheck = req.detectorCheck || null;
        if (req.body.remember) req.session.cookie.maxAge = config.session.rememberMs;
        return res.redirect('/console');
      });
    },

    logout(req, res, next) {
      req.session.destroy((err) => {
        if (err) return next(err);
        res.clearCookie(config.session.name);
        return res.redirect('/');
      });
    },
  };
}

module.exports = createAuthController;
