'use strict';

function viewLocals(protectionEnabled) {
  return (req, res, next) => {
    res.locals.protectionEnabled = protectionEnabled;
    res.locals.currentUser = (req.session && req.session.user) || null;
    next();
  };
}

module.exports = viewLocals;
