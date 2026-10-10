import katex from 'katex';

// Tokenize before Markdown so TeX braces, underscores and backslashes survive.
// Marked's fenced-code and codespan tokens retain their literal contents.
export function mathExtension() {
  const render = (text, displayMode) => katex.renderToString(text, {
    displayMode, output: 'htmlAndMathml', throwOnError: true,
    trust: false, strict: 'error', maxExpand: 1000, maxSize: 20,
    macros: { '\\sg': '\\operatorname{sg}', '\\bm': '\\boldsymbol' }
  });
  return { extensions: [
    {
      name: 'displayMath', level: 'block',
      start(src) { return src.search(/(?:^|\n) {0,3}(?:\$\$|\\\[)/); },
      tokenizer(src) {
        const match = /^(?: {0,3}\$\$[^\S\n]*\n([\s\S]+?)\n {0,3}\$\$[^\S\n]*(?:\n|$)| {0,3}\\\[[^\S\n]*\n([\s\S]+?)\n {0,3}\\\][^\S\n]*(?:\n|$))/.exec(src);
        if (match) return { type: 'displayMath', raw: match[0], text: (match[1] ?? match[2]).trim() };
      },
      renderer(token) {
        return `<div class="math-display" tabindex="0" role="region" aria-label="Equation">${render(token.text, true)}</div>\n`;
      }
    },
    {
      name: 'inlineMath', level: 'inline',
      start(src) { return src.search(/\$(?!\$|\s)|\\\(/); },
      tokenizer(src) {
        const match = /^(?:\$(?!\$|\s)((?:\\.|[^$\\\n])+?)\$(?!\d)|\\\(((?:\\(?!\))[^\n]|[^\\\n])+?)\\\))/.exec(src);
        if (match && (match[2] || !/\s$/.test(match[1]))) return { type: 'inlineMath', raw: match[0], text: match[1] ?? match[2] };
      },
      renderer(token) { return render(token.text, false); }
    }
  ] };
}
