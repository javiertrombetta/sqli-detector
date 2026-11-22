'use strict';

for (const button of document.querySelectorAll('[data-reveal]')) {
  const input = document.getElementById(button.dataset.reveal);
  button.hidden = false;
  button.addEventListener('click', () => {
    const show = input.type === 'password';
    input.type = show ? 'text' : 'password';
    button.setAttribute('aria-pressed', String(show));
    button.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  });
}
