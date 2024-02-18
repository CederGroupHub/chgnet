import { sveltekit } from '@sveltejs/kit/vite'
import * as fs from 'fs'
import type { UserConfig } from 'vite'

// fetch latest Matbench Discovery metrics table at build time and save to src/ dir
await fetch(
  `https://github.com/janosh/matbench-discovery/raw/main/site/src/figs/metrics-table-uniq-protos.svelte`,
)
  .then((res) => res.text())
  .then((text) => {
    fs.writeFileSync(`src/MetricsTable.svelte`, text)
  })

export default {
  plugins: [sveltekit()],

  server: {
    fs: { allow: [`../..`] }, // needed to import from $root
    port: 3000,
  },

  preview: {
    port: 3000,
  },
} satisfies UserConfig
