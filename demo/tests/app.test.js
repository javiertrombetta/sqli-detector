'use strict';

const test = require('node:test');
const assert = require('node:assert');
const { createApp } = require('../app');

const ATTACK = "' OR 1=1 --";
const ATTACK_LIKE_PASSWORD = "' OR '1'='1";

async function startServer(protectionEnabled) {
  const server = createApp({ protectionEnabled, logRequests: false }).listen(0);
  await new Promise((resolve) => server.once('listening', resolve));
  return { server, base: `http://127.0.0.1:${server.address().port}` };
}

function postLogin(base, fields) {
  return fetch(`${base}/login`, {
    method: 'POST',
    redirect: 'manual',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams(fields),
  });
}

function postCheck(base, text) {
  return fetch(`${base}/check`, {
    method: 'POST',
    redirect: 'manual',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ text }),
  });
}

const sessionCookie = (res) => (res.headers.get('set-cookie') || '').split(';')[0];

test('protected mode', async (t) => {
  const { server, base } = await startServer(true);
  t.after(() => server.close());

  await t.test('blocks the injection before the database is queried', async () => {
    const res = await postLogin(base, { username: ATTACK, password: 'x' });
    assert.strictEqual(res.status, 403);
    assert.match(await res.text(), /Incident reference/);
  });

  await t.test('reports the field, the score and the threshold on the block page', async () => {
    const html = await (await postLogin(base, { username: ATTACK, password: 'x' })).text();
    assert.match(html, /<th scope="row">Field<\/th><td>username<\/td>/);
    assert.match(html, /<th scope="row">Detector score<\/th><td>0\.\d{3}<\/td>/);
    assert.match(html, /<th scope="row">Threshold<\/th><td>0\.\d{3}<\/td>/);
  });

  await t.test('blocks an injection in the password of an existing user, without showing it', async () => {
    const res = await postLogin(base, { username: 'researcher', password: ATTACK });
    assert.strictEqual(res.status, 403);
    const html = await res.text();
    assert.match(html, /<th scope="row">Field<\/th><td>password<\/td>/);
    assert.doesNotMatch(html, /Strongest signals/);
  });

  await t.test('blocks a sign-in flagged by the detector even when the credentials are correct', async () => {
    const res = await postLogin(base, { username: 'operator', password: ATTACK_LIKE_PASSWORD });
    assert.strictEqual(res.status, 403);
  });

  await t.test('lets a legitimate user in and opens a session', async () => {
    const res = await postLogin(base, { username: 'researcher', password: 'researcher123' });
    assert.strictEqual(res.status, 302);
    assert.strictEqual(res.headers.get('location'), '/console');
    const page = await fetch(`${base}/console`, { headers: { cookie: sessionCookie(res) } });
    assert.strictEqual(page.status, 200);
    const html = await page.text();
    assert.match(html, /Researcher/);
    assert.match(html, /<td>username<\/td><td>0\.\d{3}<\/td>/);
    assert.match(html, /<td>password<\/td><td>0\.\d{3}<\/td>/);
  });

  await t.test('rejects a wrong password and keeps the username in the form', async () => {
    const res = await postLogin(base, { username: 'researcher', password: 'wrong' });
    assert.strictEqual(res.status, 401);
    assert.match(await res.text(), /value="researcher"/);
  });

  await t.test('does not block a name with an apostrophe', async () => {
    const res = await postLogin(base, { username: "o'neil", password: 'x' });
    assert.strictEqual(res.status, 401);
  });

  await t.test('keeps the console private', async () => {
    const res = await fetch(`${base}/console`, { redirect: 'manual' });
    assert.strictEqual(res.status, 302);
    assert.strictEqual(res.headers.get('location'), '/login');
  });

  await t.test('uses a browser-session cookie unless "Keep me signed in" is ticked', async () => {
    const plain = await postLogin(base, { username: 'researcher', password: 'researcher123' });
    assert.doesNotMatch(plain.headers.get('set-cookie'), /Expires=/i);
    const remembered = await postLogin(base, { username: 'researcher', password: 'researcher123', remember: 'on' });
    assert.match(remembered.headers.get('set-cookie'), /Expires=/i);
  });

  await t.test('signs out and ends the session', async () => {
    const login = await postLogin(base, { username: 'researcher', password: 'researcher123' });
    const cookie = sessionCookie(login);
    const logout = await fetch(`${base}/logout`, { method: 'POST', redirect: 'manual', headers: { cookie } });
    assert.strictEqual(logout.status, 302);
    const after = await fetch(`${base}/console`, { redirect: 'manual', headers: { cookie } });
    assert.strictEqual(after.status, 302);
  });

  await t.test('lists a blocked request on the console, with its reference and score', async () => {
    const blocked = await (await postLogin(base, { username: ATTACK, password: 'x' })).text();
    const reference = blocked.match(/INC-[0-9A-F]{8}/)[0];
    const login = await postLogin(base, { username: 'researcher', password: 'researcher123' });
    const html = await (await fetch(`${base}/console`, { headers: { cookie: sessionCookie(login) } })).text();
    assert.match(html, new RegExp(`<td>${reference}</td><td>username</td>`));
    assert.match(html, /<code>&#39; OR 1=1 --<\/code>/);
  });

  await t.test('never records the value of a blocked password', async () => {
    await postLogin(base, { username: 'analyst', password: ATTACK });
    const login = await postLogin(base, { username: 'researcher', password: 'researcher123' });
    const html = await (await fetch(`${base}/console`, { headers: { cookie: sessionCookie(login) } })).text();
    assert.match(html, /<td>password<\/td><td><span class="badge hold">0\.\d{3}<\/span><\/td>\s*<td>not recorded<\/td>/);
  });
});

test('public pages', async (t) => {
  const { server, base } = await startServer(true);
  t.after(() => server.close());

  await t.test('serves every public page', async () => {
    for (const path of ['/', '/login', '/css/main.css', '/js/reveal.js', '/img/lucide/shield-check.svg', '/img/lucide/LICENSE']) {
      assert.strictEqual((await fetch(`${base}${path}`)).status, 200, path);
    }
  });

  await t.test('the home page presents the project and the exported model', async () => {
    const html = await (await fetch(`${base}/`)).text();
    assert.match(html, /Protecting a Lightweight SQL Injection Detector From Unseen Obfuscation/);
    assert.match(html, /Javier Ignacio Trombetta Ghio/);
    assert.match(html, /HttpParamsDataset<\/a>, published by Morzeux in 2016 under the MIT licence/);
    assert.match(html, /<th scope="row">Decision threshold<\/th><td>0\.\d{3}/);
  });

  await t.test('the detector form scores a text and shows the decision', async () => {
    const attack = await (await postCheck(base, ATTACK)).text();
    assert.match(attack, /Flagged as SQL injection/);
    const normal = await (await postCheck(base, 'researcher')).text();
    assert.match(normal, /Not flagged/);
  });

  await t.test('an empty detector form returns to the home page', async () => {
    const res = await postCheck(base, '   ');
    assert.strictEqual(res.status, 302);
    assert.strictEqual(res.headers.get('location'), '/#try');
  });

  await t.test('the sign-in page has a close button back to the home page', async () => {
    const html = await (await fetch(`${base}/login`)).text();
    assert.match(html, /<a class="close-btn" href="\/" aria-label="Close and return to the home page"/);
  });

  await t.test('returns 404 for an unknown page', async () => {
    assert.strictEqual((await fetch(`${base}/no-such-page`)).status, 404);
  });

  await t.test('returns 413 for an oversized request', async () => {
    const res = await postLogin(base, { username: 'A'.repeat(200000), password: 'x' });
    assert.strictEqual(res.status, 413);
    assert.match(await res.text(), /<h1>Request too large<\/h1>/);
  });

  await t.test('sends security headers', async () => {
    const res = await fetch(`${base}/`);
    assert.match(res.headers.get('content-security-policy'), /default-src 'self'/);
    assert.strictEqual(res.headers.get('x-frame-options'), 'DENY');
    assert.strictEqual(res.headers.get('x-powered-by'), null);
  });
});

test('vulnerable mode', async (t) => {
  const { server, base } = await startServer(false);
  t.after(() => server.close());

  await t.test('the injection bypasses authentication', async () => {
    const res = await postLogin(base, { username: ATTACK, password: 'x' });
    assert.strictEqual(res.status, 302);
    assert.strictEqual(res.headers.get('location'), '/console');
  });

  await t.test('an account with an attack-like password signs in, and the console says it was not checked', async () => {
    const res = await postLogin(base, { username: 'operator', password: ATTACK_LIKE_PASSWORD });
    assert.strictEqual(res.status, 302);
    const page = await fetch(`${base}/console`, { headers: { cookie: sessionCookie(res) } });
    assert.match(await page.text(), /not checked by the detector/);
  });

  await t.test('the home page says that the detector is not loaded', async () => {
    const html = await (await fetch(`${base}/`)).text();
    assert.match(html, /the detector is not loaded/);
    assert.strictEqual((await postCheck(base, ATTACK)).status, 302);
  });

  await t.test('a wrong password is still rejected', async () => {
    const res = await postLogin(base, { username: 'researcher', password: 'wrong' });
    assert.strictEqual(res.status, 401);
  });

  await t.test('decodes form values exactly once, as validity_check.py assumes', async () => {
    const res = await postLogin(base, { username: 'a%2Bb', password: 'wrong' });
    assert.strictEqual(res.status, 401);
    assert.match(await res.text(), /value="a%2Bb"/);
  });

  await t.test('a UNION injection with empty columns does not break the console', async () => {
    const login = await postLogin(base, { username: "' UNION SELECT 1,'x',NULL --", password: 'x' });
    assert.strictEqual(login.status, 302);
    const res = await fetch(`${base}/console`, { headers: { cookie: sessionCookie(login) } });
    assert.strictEqual(res.status, 200);
  });

  await t.test('injected HTML is escaped on the console', async () => {
    const login = await postLogin(base, { username: "' UNION SELECT 9,'eve','<b>Eve</b>' --", password: 'x' });
    const html = await (await fetch(`${base}/console`, { headers: { cookie: sessionCookie(login) } })).text();
    assert.doesNotMatch(html, /<b>Eve<\/b>/);
    assert.match(html, /&lt;b&gt;Eve&lt;\/b&gt;/);
  });

  await t.test('a stacked DROP statement is rejected and the data survives', async () => {
    const res = await postLogin(base, { username: "'; DROP TABLE users; --", password: 'x' });
    assert.strictEqual(res.status, 401);
    const again = await postLogin(base, { username: 'researcher', password: 'researcher123' });
    assert.strictEqual(again.status, 302);
  });
});
