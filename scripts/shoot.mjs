/*
 * Screenshot harness.
 *
 * Drives the real app in a real browser and writes PNGs to screenshots/. There is no mockup
 * step and no hand-drawn image anywhere in this repo: if a screenshot shows a number, a model
 * produced that number on this machine.
 *
 *   node scripts/shoot.mjs --project p01 --url http://127.0.0.1:8101
 *   node scripts/shoot.mjs --project p02 --url http://127.0.0.1:8102
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

const PROJECT = args.project ?? 'p01';
const OUT = args.out ?? 'screenshots';
const WIDTH = Number(args.width ?? 1440);
const DEFAULT_URL = { p01: 'http://127.0.0.1:8101', p02: 'http://127.0.0.1:8102' };
const URL = args.url ?? DEFAULT_URL[PROJECT];

const RUN_TIMEOUT = 20 * 60 * 1000; // extraction on a local model is not fast

await mkdir(OUT, { recursive: true });

/** Wait for the page to settle: fonts loaded and no pending layout. */
async function settle(page) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.evaluate(() => document.fonts?.ready).catch(() => {});
  await page.waitForTimeout(250);
}

/** Submit the form and wait for results to render. */
async function submit(page, resultSelector) {
  await Promise.all([
    page.waitForURL('**/run', { timeout: RUN_TIMEOUT }),
    page.click('button[type=submit]'),
  ]);
  await page.waitForSelector(resultSelector, { timeout: RUN_TIMEOUT }).catch(() => {});
}

const SCENARIOS = {
  p01: [
    ['1-form', null],
    [
      '2-omissions-all-strategies',
      async (page) => {
        await page.selectOption('#abstract_id', 't02');
        await page.selectOption('#level', 'L4_constrained');
        await page.selectOption('#strategy', '__all__');
        await submit(page, '.fieldrow');
      },
    ],
    [
      '3-union-level',
      async (page) => {
        await page.selectOption('#abstract_id', 't09');
        await page.selectOption('#level', 'L5_union');
        await page.selectOption('#strategy', '__all__');
        await submit(page, '.fieldrow');
      },
    ],
  ],
  p02: [
    ['1-form', null],
    [
      // The headline: a narrow retrieval that withholds most of the evidence, and the model
      // answering regardless. The passage panel is what makes it legible.
      '2-phantoms-narrow-retrieval',
      async (page) => {
        await page.selectOption('#paper_id', 't01');
        await page.selectOption('#strategy', 'single_query_k4');
        await submit(page, '.fieldrow');
      },
    ],
    [
      // The control: same paper, same model, nothing withheld.
      '3-full-context-control',
      async (page) => {
        await page.selectOption('#paper_id', 't01');
        await page.selectOption('#strategy', 'full_context');
        await submit(page, '.fieldrow');
      },
    ],
    [
      // The fix: per-field queries, a third of the context, nearly the control's accuracy.
      '4-per-field-fix',
      async (page) => {
        await page.selectOption('#paper_id', 't01');
        await page.selectOption('#strategy', 'per_field_n1');
        await submit(page, '.fieldrow');
      },
    ],
  ],
};

const scenarios = SCENARIOS[PROJECT];
if (!scenarios) {
  console.error(`unknown project: ${PROJECT}`);
  process.exit(1);
}

// `channel: 'chromium'` uses the full browser build rather than the headless shell, which is
// what `npx playwright install chromium` actually fetches.
const browser = await chromium.launch({ channel: 'chromium' });

try {
  for (const scheme of ['light', 'dark']) {
    console.log(`${scheme}:`);
    for (const [name, steps] of scenarios) {
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
      await settle(page);
      const file = path.join(OUT, `${PROJECT}-${name}-${scheme}.png`);
      await page.screenshot({ path: file, fullPage: true });
      console.log(`  wrote ${file}`);
      await context.close();
    }
  }
} finally {
  await browser.close();
}

console.log('done');
