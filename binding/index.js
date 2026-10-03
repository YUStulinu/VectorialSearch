'use strict';
// The addon lands in one of these depending on how it was built.
const path = require('path');
let native;
const candidates = [
  '../build/Release/vecsearch.node',
  '../build/Debug/vecsearch.node'
];
let lastError;
for (const rel of candidates) {
  try { native = require(path.join(__dirname, rel)); break; }
  catch (err) { lastError = err; }
}
if (!native) {
  throw new Error(
    'The vecsearch native addon is not built.\n' +
    'Run:  npm run rebuild\n' +
    `(last error: ${lastError && lastError.message})`
  );
}
module.exports = native;
