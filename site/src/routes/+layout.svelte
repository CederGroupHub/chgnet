<script lang="ts">
  import { goto } from '$app/navigation'
  import { page } from '$app/stores'
  import { repository } from '$site/package.json'
  import { CmdPalette } from 'svelte-multiselect'
  import Toc from 'svelte-toc'
  import { CopyButton, GitHubCorner } from 'svelte-zoo'
  import '../app.css'

  $: headingSelector = `main :is(${
    $page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`
  }):not(.toc-exclude)`

  const file_routes = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`))
    .filter((key) => !key.includes(`/[`))
    .map((filename) => {
      const parts = filename.split(`/`)
      return `/` + parts.slice(1, -1).join(`/`)
    })

  const notebooks = Object.keys(
    import.meta.glob(`$root/examples/*.html`, { eager: true, as: `url` })
  ).map((path) => {
    const filename = path.split(`/`).at(-1)?.replace(`.html`, ``)
    return `/notebooks/${filename}`
  })
  const actions = file_routes.concat(notebooks).map((name) => {
    return { label: name, action: () => goto(name.toLowerCase()) }
  })
</script>

<CmdPalette {actions} placeholder="Go to..." />

<CopyButton global />

<Toc {headingSelector} breakpoint={1250} warnOnEmpty={false} />

{#if $page.url.pathname !== `/`}
  <a href="/" aria-label="Back to index page">&laquo; home</a>
{/if}

<GitHubCorner href={repository} />

<slot />

<footer>
  Questions/feedback?
  <a href="{repository}/issues">GitHub issues</a>
  <a href="mailto:bdeng@lbl.gov">email</a>
</footer>

<style>
  a[href='/'] {
    font-size: 15pt;
    position: absolute;
    top: 2em;
    left: 2em;
    background-color: rgba(255, 255, 255, 0.1);
    padding: 1pt 5pt;
    border-radius: 3pt;
    transition: 0.2s;
  }
  a[href='/']:hover {
    background-color: rgba(255, 255, 255, 0.2);
  }
  footer {
    display: flex;
    gap: 1ex;
    place-content: center;
    place-items: center;
    margin: 2em 0 0;
    padding: 3vh 3vw;
    background: #00061a;
  }
</style>
