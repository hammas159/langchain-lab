/*
 * Screenshot harness.
 *
 * Drives the real app in a real browser and writes PNGs to screenshots/. There is no mockup
 * step and no hand-drawn image anywhere in this repo: if a screenshot shows a number, a model
 * produced that number on this machine.
 *
 *   node scripts/shoot.mjs --url http://127.0.0.1:8101 --out screenshots --prefix p01
 *
 * Each shot is taken twice, once in each colour scheme, because the design system defines
 * both and a dark-mode bug is invisible if you only ever screenshot in light.
 */

import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, cur, i, arr) => {
    if (cur.startsWith('--')) acc.push([cur.slice(2), arr[i + 1]]);
    return acc;
  }, []),
);

const URL = args.url ?? 'http://127.0.0.1:8101';
const OUT = args.out ?? 'screenshots';
const PREFIX = args.prefix ?? 'shot';
const WIDTH = Number(args.width ?? 1440);

await mkdir(OUT, { recursive: true });

/** Wait for the page to settle: fonts loaded and no pending layout. */
async function settle(page) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.evaluate(() => document.fonts?.ready).catch(() => {});
  await page.waitForTimeout(250);
}

async function shoot(page, name, scheme) {
  await settle(page);
  const file = path.join(OUT, `${PREFIX}-${name}-${scheme}.png`);
  await page.screenshot({ path: file, fullPage: true });
  console.log(`  wrote ${file}`);
}

/**
 * One scripted visit. `steps` receives the page and performs the interaction; whatever it
 * leaves on screen is what gets captured.
 */
async function visit(browser, scheme, name, steps) {
  const context = await browser.newContext({
    colorScheme: scheme,
    // Deliberately short. `fullPage` grows to fit the content but never shrinks below the
    // viewport, so a tall viewport pads every screenshot of a short page with dead space.
    viewport: { width: WIDTH, height: 700 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();
  page.on('pageerror', (e) => console.error(`  ! page error: ${e.message}`));
  await page.goto(URL, { waitUntil: 'domcontentloaded' });
  if (steps) await steps(page);
  await shoot(page, name, scheme);
  await context.close();
}

const RUN_TIMEOUT = 15 * 60 * 1000; // extraction on a local model is not fast

/** Fill the form and wait for the results to come back. */
function runExtraction({ abstract, level, strategy }) {
  return async (page) => {
    await page.selectOption('#abstract_id', abstract);
    await page.selectOption('#level', level);
    await page.selectOption('#strategy', strategy);
    await Promise.all([
      page.waitForURL('**/run', { timeout: RUN_TIMEOUT }),
      page.click('button[type=submit]'),
    ]);
    await page.waitForSelector('.fieldrow', { timeout: RUN_TIMEOUT }).catch(() => {});
  };
}

// `channel: 'chromium'` uses the full browser build rather than the headless shell, which is
// the one `npx playwright install chromium` actually fetches.
const browser = await chromium.launch({ channel: 'chromium' });

try {
  for (const scheme of ['light', 'dark']) {
    console.log(`${scheme}:`);
    await visit(browser, scheme, '1-form', null);
    await visit(
      browser,
      scheme,
      '2-omissions-all-strategies',
      runExtraction({ abstract: 't02', level: 'L4_constrained', strategy: '__all__' }),
    );
    await visit(
      browser,
      scheme,
      '3-union-level',
      runExtraction({ abstract: 't09', level: 'L5_union', strategy: '__all__' }),
    );
  }
} finally {
  await browser.close();
}

console.log('done');
