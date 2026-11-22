'use strict';

function formText(value) {
  if (Array.isArray(value)) return value.join(',');
  return value === undefined || value === null ? '' : String(value);
}

function showSpaces(text) {
  return String(text).replace(/ /g, '\u2423');
}

module.exports = { formText, showSpaces };
