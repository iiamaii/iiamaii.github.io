import test from 'node:test';
import assert from 'node:assert/strict';
import { Marked } from 'marked';
import { mathExtension } from './math.mjs';

const parse = source => new Marked(mathExtension()).parse(source);
test('TeX survives Markdown in prose, lists and tables with accessible MathML', () => {
  const html = parse('State $z^{[i]}_{1:T}$ and \\(\\mathcal{C}^{s}\\).\n\n- $x_i^2$\n\n| Symbol | Meaning |\n| --- | --- |\n| $\\lambda$ | weight |');
  assert.equal((html.match(/class="katex"/g) || []).length, 4);
  assert.match(html, /<math /);
  assert.match(html, /application\/x-tex/);
  assert.doesNotMatch(html, /<em>/);
});
test('display blocks preserve aligned equations and comparison signs', () => {
  for (const [open, close] of [['$$','$$'], ['\\[','\\]']]) {
    const html = parse(`${open}\n\\begin{aligned}z^\\star &= f_\\theta(z^\\star;x)\\\\ \\|J\\|_2 &< 1\\end{aligned}\n${close}`);
    assert.match(html, /class="math-display"/);
    assert.match(html, /class="katex-display"/);
    assert.doesNotMatch(html, /<pre>/);
  }
});
test('code, escaped dollars and currency retain literal meaning', () => {
  const html = parse('`$x_i$`\n\n```python\nprint("$x_i$")\n```\n\nCost \\$5 or $10.');
  assert.doesNotMatch(html, /class="katex"/);
  assert.match(html, /<code>\$x_i\$<\/code>/);
});
test('invalid TeX fails the build; unsafe commands cannot create links', () => {
  assert.throws(() => parse('$\\notARealCommand{x}$'), /Undefined control sequence/);
  const html = parse('$\\href{javascript:alert(1)}{x}$');
  assert.doesNotMatch(html, /href=/);
});
