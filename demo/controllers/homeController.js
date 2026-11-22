'use strict';

const project = require('../config/project');
const { formText } = require('../helpers/text');

const MAX_CHECK_LENGTH = 300;

function createHomeController({ protectionEnabled }) {
  const detector = protectionEnabled ? require('../security/detector') : null;
  const model = detector && detector.info;

  const render = (res, check = null) =>
    res.render('home', { title: `${project.siteName} | ${project.title}`, model, check });

  return {
    index(req, res) {
      render(res);
    },

    check(req, res) {
      const text = formText(req.body.text).slice(0, MAX_CHECK_LENGTH);
      if (!detector || !text.trim()) return res.redirect('/#try');
      return render(res, { text, ...detector.inspect(text) });
    },
  };
}

module.exports = createHomeController;
