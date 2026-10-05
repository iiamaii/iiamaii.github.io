import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const result = spawnSync(process.execPath, ['scripts/build.mjs'], { stdio: 'inherit' });
if (result.status !== 0) process.exit(result.status ?? 1);
const root = path.resolve('dist');
const config = JSON.parse(await fs.readFile('site.json', 'utf8'));
const base = (process.env.SITE_BASE_PATH ?? config.basePath ?? '').replace(/\/$/, '');
const port = Number(process.env.PORT ?? 4321);
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.webp': 'image/webp', '.png': 'image/png', '.xml': 'application/xml; charset=utf-8', '.txt': 'text/plain; charset=utf-8', '.woff2': 'font/woff2' };
http.createServer(async (req, res) => {
  try {
    let pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (base && !pathname.startsWith(`${base}/`) && pathname !== base) throw new Error('not-found');
    if (base) pathname = pathname.slice(base.length) || '/';
    let file = path.resolve(root, `.${pathname}`);
    if (file !== root && !file.startsWith(`${root}${path.sep}`)) throw new Error('not-found');
    if ((await fs.stat(file)).isDirectory()) file = path.join(file, 'index.html');
    const data = await fs.readFile(file);
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] ?? 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(data);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(await fs.readFile(path.join(root, '404.html')));
  }
}).listen(port, '127.0.0.1', () => console.log(`Preview: http://localhost:${port}${base}/ (run npm run build after edits)`));
