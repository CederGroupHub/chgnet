<script lang="ts">
  import Readme from '$root/README.md'
  import MetricsTable from '$src/MetricsTable.svelte'
</script>

<main>
  <Readme>
    <MetricsTable slot="metrics-table" />
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
  :global(a:has(img[alt='Docs'])) {
    display: none;
  }
</style>
