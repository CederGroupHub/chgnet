<main>
  <h1 class="toc-exclude">API</h1>

  {#each Object.values(import.meta.glob(`./*.md`, { eager: true })) as file}
    <svelte:component this={file?.default} />
  {/each}
</main>

<style>
  /* select all but first module h1s */
  :global(h1[id^='module-']:not(:nth-of-type(2))) {
    margin: 4em 0 0;
  }
  :global(hr) {
    border: none;
  }
</style>
