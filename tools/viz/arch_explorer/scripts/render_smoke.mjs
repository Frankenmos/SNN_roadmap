// Headless render smoke test: serves the built app, opens it in a
// headless system browser (Edge/Chrome via puppeteer-core - no browser
// download), waits for the scene to report real rendered frames, and
// saves a screenshot. Fails if the canvas never draws.
//
// Two phases:
//   A. static mode - no run_data.json (fetch 404 must degrade cleanly)
//   B. live mode  - a synthetic run_data.json is written into dist/ and
//      the info panel must show its learned constants / action mix.
// The preview server is restarted between phases because sirv builds
// its file map at startup.
//
// Usage: npm run build && npm run smoke
// Override the browser: set ARCH_EXPLORER_BROWSER=<path to exe>

import { existsSync } from 'node:fs'
import { stat, unlink, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { preview } from 'vite'
import puppeteer from 'puppeteer-core'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const PORT = 4173

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function findBrowser() {
  const candidates = [
    process.env.ARCH_EXPLORER_BROWSER,
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    `${process.env.LOCALAPPDATA ?? ''}/Google/Chrome/Application/chrome.exe`,
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
    '/usr/bin/chromium',
  ]
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) return candidate
  }
  throw new Error(
    'No Edge/Chrome found. Set ARCH_EXPLORER_BROWSER to a browser executable.',
  )
}

// Synthetic bundle matching tools/registry export schema_version 1.
// Values are chosen so the assertions below are unambiguous strings.
const range = Array.from({ length: 12 }, (_, i) => i + 1)
const SMOKE_RUN_DATA = {
  schema_version: 1,
  kind: 'arch-explorer-run-data',
  run: 'smoke_run',
  generated_iso: '2026-07-06T00:00:00Z',
  config: {
    reward_name: 'defeat_roaches_v4',
    spatial_head_type: 'coarse_to_fine',
    lr: 5e-5,
    sil_enabled: true,
    snn_init: {
      fast_alpha: 0.55,
      fast_beta: 0.65,
      slow_alpha: 0.92,
      slow_beta: 0.97,
    },
  },
  module_param_counts: {
    entity_encoder: 72136,
    target_head: 51408,
    shared_fc1: 41088,
  },
  entries: [
    {
      ref: 'smoke_run:u5',
      file: 'policy_u5.pth',
      kind: 'snapshot',
      policy_version: 5,
      episode: 50,
      wall_time_iso: '2026-07-05T10:00:00',
      git_commit: 'deadbeef',
      size_mib: 4.2,
      eval_mean: 1.5,
      eval_policy_version: 5,
      time_constants: [
        { name: 'token_snn.snn.alpha', kind: 'alpha', mean: 0.58, effective_mean: 0.58 },
        { name: 'token_snn.snn.beta', kind: 'beta', mean: 0.66, effective_mean: 0.66 },
        { name: 'slow_token_snn.snn.alpha', kind: 'alpha', mean: 0.95, effective_mean: 0.95 },
        { name: 'slow_token_snn.snn.beta', kind: 'beta', mean: 0.98, effective_mean: 0.98 },
      ],
      update_row: {
        global_update_index: 5,
        mean_entropy: 0.9,
        mean_kl: 0.01,
        clip_fraction: 0.1,
        grad_norm: 1.2,
        rollout_policy_no_op_count: 2000,
        rollout_policy_left_click_count: 0,
        rollout_policy_right_click_count: 150,
        rollout_feedback_near_enemy_smart_count: 40,
        rollout_feedback_enemy_health_drop_after_smart_count: 12,
      },
    },
    {
      ref: 'smoke_run:checkpoint',
      file: 'checkpoint.pth',
      kind: 'checkpoint',
      policy_version: 12,
      episode: 120,
      wall_time_iso: '2026-07-06T10:00:00',
      git_commit: 'deadbeef',
      size_mib: 4.2,
      eval_mean: 9.0,
      eval_policy_version: 11,
      time_constants: [
        { name: 'token_snn.snn.alpha', kind: 'alpha', mean: 0.6128, effective_mean: 0.6128 },
        { name: 'token_snn.snn.beta', kind: 'beta', mean: 0.7123, effective_mean: 0.7123 },
        { name: 'slow_token_snn.snn.alpha', kind: 'alpha', mean: 1.0007, effective_mean: 1.0 },
        { name: 'slow_token_snn.snn.beta', kind: 'beta', mean: 1.0004, effective_mean: 1.0 },
      ],
      update_row: {
        global_update_index: 12,
        mean_entropy: 0.7,
        mean_kl: 0.012,
        clip_fraction: 0.08,
        grad_norm: 0.9,
        grad_norm_trunk: 0.5,
        grad_norm_actor_head: 0.2,
        grad_norm_critic_head: 0.15,
        grad_norm_target_head: 0.05,
        rollout_policy_no_op_count: 36,
        rollout_policy_left_click_count: 0,
        rollout_policy_right_click_count: 3128,
        rollout_feedback_near_enemy_smart_count: 900,
        rollout_feedback_enemy_health_drop_after_smart_count: 420,
        sil_loss: 0.03,
        sil_gate_open_fraction: 0.4,
        sil_buffer_size: 5000,
        sil_steps_replayed: 512,
      },
    },
  ],
  history: {
    total_updates: 12,
    stride: 1,
    series: {
      global_update_index: range,
      mean_entropy: range.map((i) => 1 - i * 0.02),
      grad_norm: range.map((i) => 1.5 - i * 0.05),
      rollout_policy_no_op_count: range.map((i) => 3000 - i * 240),
      rollout_policy_left_click_count: range.map(() => 0),
      rollout_policy_right_click_count: range.map((i) => 100 + i * 250),
      sil_gate_open_fraction: range.map((i) => 0.1 + i * 0.02),
    },
  },
  evals: [],
}

// Case-insensitive: CSS text-transform (uppercase chips/headers) changes
// what innerText reports.
async function bodyIncludes(page, needle) {
  return page.evaluate(
    (text) => document.body.innerText.toLowerCase().includes(text),
    needle.toLowerCase(),
  )
}

async function main() {
  if (!existsSync(join(root, 'dist', 'index.html'))) {
    throw new Error('dist/index.html missing - run `npm run build` first.')
  }
  const runDataPath = join(root, 'dist', 'run_data.json')
  if (existsSync(runDataPath)) await unlink(runDataPath) // stale bundle

  const previewOptions = {
    root,
    preview: { port: PORT, strictPort: true, open: false },
  }
  let server = await preview(previewOptions)
  const browserPath = findBrowser()
  console.log(`[smoke] serving dist on :${PORT}, browser: ${browserPath}`)

  const browser = await puppeteer.launch({
    executablePath: browserPath,
    headless: true,
    args: [
      '--no-sandbox',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--window-size=1400,900',
    ],
  })

  try {
    // ------------------------------------------------ phase A: static
    const page = await browser.newPage()
    await page.setViewport({ width: 1400, height: 900 })
    const errors = []
    page.on('pageerror', (error) => errors.push(String(error)))
    await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' })

    // The scene sets this after 8 real rendered frames.
    await page.waitForFunction('window.__ARCH_EXPLORER_READY === true', {
      timeout: 30_000,
    })

    const canvasInfo = await page.evaluate(() => {
      const canvas = document.querySelector('canvas')
      return canvas
        ? { width: canvas.width, height: canvas.height }
        : null
    })
    if (!canvasInfo || canvasInfo.width === 0) {
      throw new Error('WebGL canvas missing or zero-sized.')
    }

    // Resize handling: the canvas must track the viewport.
    await page.setViewport({ width: 1000, height: 700 })
    await sleep(600)
    const resized = await page.evaluate(() => {
      const canvas = document.querySelector('canvas')
      return canvas ? canvas.clientWidth : 0
    })
    if (Math.abs(resized - 1000) > 40) {
      throw new Error(`Canvas did not follow resize (clientWidth=${resized}).`)
    }
    await page.setViewport({ width: 1400, height: 900 })
    await sleep(400)

    // Interactivity path: select the attention zone via the debug hook
    // and check the info panel shows the REAL source excerpt.
    await page.evaluate(() => window.__ARCH_EXPLORER_SELECT('attention'))
    await sleep(1200)
    if (!(await bodyIncludes(page, 'scaled_dot_product_attention'))) {
      throw new Error('Info panel did not render the real code excerpt.')
    }
    // Without a bundle there must be no live badge.
    if (await bodyIncludes(page, 'live run data')) {
      throw new Error('Live badge shown without a run_data.json bundle.')
    }

    const shotPath = join(root, 'smoke_screenshot.png')
    await page.screenshot({ path: shotPath })
    const { size } = await stat(shotPath)
    // A blank dark page compresses to a few KB; the neon scene does not.
    if (size < 30_000) {
      throw new Error(
        `Screenshot suspiciously small (${size} bytes) - canvas likely blank.`,
      )
    }
    if (errors.length) {
      throw new Error(`Page errors (static): ${errors.join(' | ')}`)
    }

    // ------------------------------------------------ phase B: live
    // Restart the server so sirv indexes the new file, then reload.
    await new Promise((resolve) => server.httpServer.close(resolve))
    await writeFile(runDataPath, JSON.stringify(SMOKE_RUN_DATA))
    server = await preview(previewOptions)
    await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' })
    await page.waitForFunction('window.__ARCH_EXPLORER_READY === true', {
      timeout: 30_000,
    })
    await sleep(400)

    if (!(await bodyIncludes(page, 'live run data'))) {
      throw new Error('Live badge missing with run_data.json present.')
    }
    if (!(await bodyIncludes(page, 'smoke_run'))) {
      throw new Error('Live badge does not show the run name.')
    }

    await page.evaluate(() => window.__ARCH_EXPLORER_SELECT('snn'))
    await sleep(1200)
    if (!(await bodyIncludes(page, '0.7123'))) {
      throw new Error('SNN live section missing the learned fast beta.')
    }
    if (!(await bodyIncludes(page, 'no-leak integrator'))) {
      throw new Error('SNN live section missing the clamped-beta note.')
    }

    await page.evaluate(() => window.__ARCH_EXPLORER_SELECT('dispatch'))
    await sleep(800)
    if (!(await bodyIncludes(page, '3,128'))) {
      throw new Error('Dispatch live section missing the right-click count.')
    }

    const liveShotPath = join(root, 'smoke_screenshot_live.png')
    await page.screenshot({ path: liveShotPath })
    if (errors.length) {
      throw new Error(`Page errors (live): ${errors.join(' | ')}`)
    }

    console.log(
      `[smoke] PASS - canvas ${canvasInfo.width}x${canvasInfo.height}, ` +
        `resize OK, live mode OK, screenshots ${Math.round(size / 1024)} KiB ` +
        `-> ${shotPath} + ${liveShotPath}`,
    )
  } finally {
    await browser.close()
    await new Promise((resolve) => server.httpServer.close(resolve))
    if (existsSync(runDataPath)) await unlink(runDataPath)
  }
}

main().catch((error) => {
  console.error(`[smoke] FAIL - ${error.message}`)
  process.exit(1)
})
