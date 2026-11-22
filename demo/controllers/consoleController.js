'use strict';

const project = require('../config/project');
const { clockTime } = require('../helpers/time');

function createConsoleController({ incidents, protectionEnabled }) {
  const detector = protectionEnabled ? require('../security/detector') : null;
  const model = detector && detector.info;

  return {
    index(req, res) {
      res.render('console/index', {
        title: `Console | ${project.siteName}`,
        model,
        detectorCheck: req.session.detectorCheck || null,
        scored: incidents.scored,
        blockedTotal: incidents.blockedTotal,
        blocked: incidents.blocked.map((entry) => ({ ...entry, time: clockTime(entry.time) })),
      });
    },
  };
}

module.exports = createConsoleController;
