'use strict';

module.exports = {
  siteName: 'SQLi Detector',
  title: 'Protecting a Lightweight SQL Injection Detector From Unseen Obfuscation',
  author: 'Javier Ignacio Trombetta Ghio',
  repository: 'https://github.com/javiertrombetta/sqli-detector',

  dataset: {
    name: 'HttpParamsDataset',
    author: 'Morzeux',
    year: 2016,
    licence: 'MIT',
    url: 'https://github.com/Morzeux/HttpParamsDataset',
    train: 21109,
    val: 4523,
    test: 4524,
  },

  cleanTest: { detection: 99.9, falsePositives: 0.97, f1: 0.991 },

  unseenEncodings: [
    { version: 'A', defence: 'No defence', detected: 89.6, normalFlagged: 54.8 },
    { version: 'B', defence: 'Adversarial training', detected: 98.1, normalFlagged: 57.5 },
    { version: 'C', defence: 'Input normalisation', detected: 89.6, normalFlagged: 22.0 },
    { version: 'D', defence: 'Both defences', detected: 95.7, normalFlagged: 22.4 },
  ],
};
