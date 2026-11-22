'use strict';

const path = require('path');
const express = require('express');
const session = require('express-session');

const config = require('./config');
const project = require('./config/project');
const { showSpaces } = require('./helpers/text');
const { createDatabase } = require('./models/database');
const UserModel = require('./models/userModel');
const IncidentLog = require('./models/incidentLog');
const createHomeController = require('./controllers/homeController');
const createAuthController = require('./controllers/authController');
const createConsoleController = require('./controllers/consoleController');
const errors = require('./controllers/errorController');
const createRouter = require('./routes');
const securityHeaders = require('./middleware/securityHeaders');
const requestLogger = require('./middleware/requestLogger');
const viewLocals = require('./middleware/viewLocals');

function createApp({ protectionEnabled = config.protectionEnabled, logRequests = true } = {}) {
  const db = createDatabase();
  const users = new UserModel(db);
  const incidents = new IncidentLog();

  const app = express();
  app.set('views', path.join(__dirname, 'views'));
  app.set('view engine', 'ejs');
  app.disable('x-powered-by');
  app.locals.project = project;
  app.locals.showSpaces = showSpaces;

  if (logRequests) app.use(requestLogger);
  app.use(securityHeaders);
  app.use(express.static(path.join(__dirname, 'public')));
  app.use(express.urlencoded({ extended: false }));
  app.use(session({
    name: config.session.name,
    secret: config.session.secret,
    resave: false,
    saveUninitialized: false,
    cookie: { httpOnly: true, sameSite: 'lax' },
  }));
  app.use(viewLocals(protectionEnabled));

  app.use(createRouter({
    home: createHomeController({ protectionEnabled }),
    auth: createAuthController({ users, protectionEnabled }),
    consolePage: createConsoleController({ incidents, protectionEnabled }),
    incidents,
    protectionEnabled,
  }));

  app.use(errors.notFound);
  app.use(errors.serverError);
  return app;
}

if (require.main === module) {
  const server = createApp().listen(config.port, () => {
    console.log(`SQLi Detector demonstration running at http://localhost:${config.port} ` +
                `(login protection: ${config.protectionEnabled ? 'ON' : 'OFF'})`);
  });
  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      console.error(`Port ${config.port} is already in use. Stop the other server (Ctrl + C in its ` +
                    'terminal) or set the PORT environment variable to a free port.');
    } else {
      console.error(err);
    }
    process.exit(1);
  });
}

module.exports = { createApp };
