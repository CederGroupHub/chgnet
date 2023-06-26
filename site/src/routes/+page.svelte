<script lang="ts">
  import Readme from '$root/README.md'
</script>

<main>
  <Readme>
    <svelte:fragment slot="docs">
      <h2>Docs</h2>
      View&nbsp;<a href="/api">CHGNet API docs</a>
    </svelte:fragment>
  </Readme>
</main>

<style>
  :global(h1[align='center']) {
    display: flex;
    font-size: clamp(2rem, 2rem + 2vw, 3rem);
    place-items: center;
    place-content: center;
    gap: 6pt;
  }
</style>
