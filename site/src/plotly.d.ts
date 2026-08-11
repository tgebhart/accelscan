// `plotly.js-dist-min` ships no type declarations. This is a minimal, loosely
// typed shim covering only the calls this site makes (newPlot).
declare module 'plotly.js-dist-min' {
  const Plotly: {
    newPlot: (
      root: string | HTMLElement,
      data: unknown[],
      layout?: Record<string, unknown>,
      config?: Record<string, unknown>,
    ) => Promise<unknown>;
  };
  export default Plotly;
}
