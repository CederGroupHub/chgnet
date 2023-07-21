<script lang="ts">
  import { goto } from '$app/navigation'
  import { base } from '$app/paths'
  import { page } from '$app/stores'
  import { repository } from '$site/package.json'
  import { CmdPalette } from 'svelte-multiselect'
  import Toc from 'svelte-toc'
  import { CopyButton, GitHubCorner } from 'svelte-zoo'
  import '../app.css'

  $: headingSelector = `main :is(${
    $page.url.pathname === `${base}/api` ? `h1, h2, h3, h4` : `h2`
  }):not(.toc-exclude)`

  const file_routes = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`))
    .filter((key) => !key.includes(`/[`))
    .map((filename) => {
      const parts = filename.split(`/`)
      return `/` + parts.slice(1, -1).join(`/`)
    })

  const actions = file_routes.map((name) => {
    return { label: name, action: () => goto(`${base}${name.toLowerCase()}`) }
  })
</script>

<CmdPalette {actions} placeholder="Go to..." />

<CopyButton global />

<Toc {headingSelector} breakpoint={1250} warnOnEmpty={false} />

{#if ![base, `/`].includes($page.url.pathname)}
  <a href="{base}/" aria-label="Back to index page">&laquo; home</a>
{/if}

<GitHubCorner href={repository} />

<slot />

<footer>
  <img
    src="https://raw.github.com/CederGroupHub/chgnet/main/site/static/chgnet-logo.png"
    alt="Logo"
    width="300px"
  />
  <a href="{repository}/issues" rel="external">Issues</a>
  <a href="{repository}/discussions" rel="external">Discussions</a>
  <a href="/api">API</a>
</footer>

<style>
  footer {
    background: #00061a;
    display: flex;
    flex-wrap: wrap;
    gap: 3ex;
    place-content: center;
    place-items: center;
    margin: 2em 0 0;
    padding: 3vh 3vw;
  }
  a[aria-label] {
    font-size: 15pt;
    position: absolute;
    top: 2em;
    left: 2em;
    background-color: rgba(255, 255, 255, 0.1);
    padding: 1pt 5pt;
    border-radius: 3pt;
    transition: 0.2s;
  }
  a[aria-label]:hover {
    background-color: rgba(255, 255, 255, 0.2);
  }
</style>
