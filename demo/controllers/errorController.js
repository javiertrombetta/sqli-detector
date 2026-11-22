'use strict';

const project = require('../config/project');

const MESSAGES = {
  413: { heading: 'Request too large', text: 'The information you sent is larger than the site accepts. Shorten it and try again.' },
  400: { heading: 'Request not understood', text: 'The site could not read your request. Go back and try again.' },
};

exports.notFound = (req, res) => {
  res.status(404).render('errors/404', { title: `Page not found | ${project.siteName}` });
};

exports.serverError = (err, req, res, next) => {
  if (res.headersSent) return next(err);
  const status = Number(err.status || err.statusCode) || 500;
  if (status >= 500) console.error(`[error] ${err.stack || err}`);
  else console.warn(`[client error] ${status} ${err.message}`);

  const message = MESSAGES[status] || {
    heading: 'Something went wrong',
    text: 'The site could not complete your request. Go back and try again.',
  };
  res.status(status).render('errors/error', { title: `${message.heading} | ${project.siteName}`, ...message });
};
