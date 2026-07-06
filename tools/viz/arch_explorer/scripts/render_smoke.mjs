// Headless render smoke test: serves the built app, opens it in a
// headless system browser (Edge/Chrome via puppeteer-core - no browser
// download), waits for the scene to report real rendered frames, and
// saves a screenshot. Fails if the canvas never draws.
//
// Usage: npm run build && npm run smoke
// Override the browser: set ARCH_EXPLORER_BROWSER=<path to exe>

import { existsSync } from 'node:fs'
import { stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { preview } from 'vite'
import puppeteer from 'puppeteer-core'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const PORT = 4173

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

async function main() {
  if (!existsSync(join(root, 'dist', 'index.html'))) {
    throw new Error('dist/index.html missing - run `npm run build` first.')
  }

  const server = await preview({
    root,
    preview: { port: PORT, strictPort: true, open: false },
  })
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
    await new Promise((resolve) => setTimeout(resolve, 600))
    const resized = await page.evaluate(() => {
      const canvas = document.querySelector('canvas')
      return canvas ? canvas.clientWidth : 0
    })
    if (Math.abs(resized - 1000) > 40) {
      throw new Error(`Canvas did not follow resize (clientWidth=${resized}).`)
    }
    await page.setViewport({ width: 1400, height: 900 })
    await new Promise((resolve) => setTimeout(resolve, 400))

    // Interactivity path: select the attention zone via the debug hook
    // and check the info panel shows the REAL source excerpt.
    await page.evaluate(() => window.__ARCH_EXPLORER_SELECT('attention'))
    await new Promise((resolve) => setTimeout(resolve, 1200))
    const panelHasRealCode = await page.evaluate(() =>
      document.body.innerText.includes('scaled_dot_product_attention'),
    )
    if (!panelHasRealCode) {
      throw new Error('Info panel did not render the real code excerpt.')
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
      throw new Error(`Page errors: ${errors.join(' | ')}`)
    }

    console.log(
      `[smoke] PASS - canvas ${canvasInfo.width}x${canvasInfo.height}, ` +
        `resize OK, screenshot ${Math.round(size / 1024)} KiB -> ${shotPath}`,
    )
  } finally {
    await browser.close()
    await new Promise((resolve) => server.httpServer.close(resolve))
  }
}

main().catch((error) => {
  console.error(`[smoke] FAIL - ${error.message}`)
  process.exit(1)
})
