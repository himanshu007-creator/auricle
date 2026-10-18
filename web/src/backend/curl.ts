// Reads curl commands pasted by the user and fills backend settings. Nothing is
// executed; this is a tokenizer plus a few rules.

export interface ParsedCurl {
  baseURL?: string;
  apiKey?: string;
  headers: Record<string, string>;
  models: { stt?: string; llm?: string; tts?: string };
  voice?: string;
  warnings: string[];
}

type Role = 'stt' | 'llm' | 'tts';

const ROLE_PATHS: [RegExp, Role][] = [
  [/\/chat\/completions\/?$/, 'llm'],
  [/\/audio\/transcriptions\/?$/, 'stt'],
  [/\/audio\/speech\/?$/, 'tts'],
];

/** Shell-style tokenizer: joins backslash continuations, handles single and double quotes. */
export function tokenize(command: string): string[] {
  const text = command.replace(/\\\r?\n/g, ' ');
  const tokens: string[] = [];
  let current = '';
  let quote: '"' | "'" | null = null;
  let hasToken = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === quote) quote = null;
      else if (ch === '\\' && quote === '"' && i + 1 < text.length) current += text[++i];
      else current += ch;
      hasToken = true;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
      hasToken = true;
    } else if (ch === '\\' && i + 1 < text.length) {
      current += text[++i];
      hasToken = true;
    } else if (/\s/.test(ch)) {
      if (hasToken) tokens.push(current);
      current = '';
      hasToken = false;
    } else {
      current += ch;
      hasToken = true;
    }
  }
  if (hasToken) tokens.push(current);
  return tokens;
}

/** Splits pasted text into one string per curl command. */
export function splitCommands(text: string): string[] {
  const joined = text.replace(/\\\r?\n/g, ' ');
  const parts = joined.split(/(?:^|\n)\s*(?=curl\s)/).map((s) => s.trim()).filter(Boolean);
  return parts.filter((p) => p.startsWith('curl'));
}

export function parseCurl(text: string): ParsedCurl {
  const result: ParsedCurl = { headers: {}, models: {}, warnings: [] };
  const commands = splitCommands(text);
  if (commands.length === 0) result.warnings.push('No curl command found.');
  for (const command of commands) parseOne(tokenize(command), result);
  return result;
}

function parseOne(tokens: string[], result: ParsedCurl): void {
  let url: string | undefined;
  let body: string | undefined;
  const headers: Record<string, string> = {};
  const formFields: Record<string, string> = {};
  for (let i = 1; i < tokens.length; i++) {
    const t = tokens[i];
    const next = () => tokens[++i] ?? '';
    if (t === '-H' || t === '--header') {
      const [name, ...rest] = next().split(':');
      headers[name.trim().toLowerCase()] = rest.join(':').trim();
    } else if (t === '-d' || t === '--data' || t === '--data-raw' || t === '--data-binary' || t === '--json') {
      body = next();
    } else if (t === '-F' || t === '--form') {
      const [name, ...rest] = next().split('=');
      formFields[name.trim()] = rest.join('=');
    } else if (t === '-X' || t === '--request' || t === '-o' || t === '--output' || t === '-u' || t === '--user') {
      next();
    } else if (t === '--url') {
      url = next();
    } else if (/^https?:\/\//.test(t) && !url) {
      url = t;
    }
  }
  if (!url) {
    result.warnings.push('A command had no http(s) URL.');
    return;
  }

  let role: Role | undefined;
  for (const [pattern, r] of ROLE_PATHS) {
    if (pattern.test(url)) {
      role = r;
      result.baseURL = url.replace(pattern, '');
      break;
    }
  }
  if (!role) {
    result.warnings.push(`Could not tell which role ${url} serves; using it as the base URL.`);
    result.baseURL = url.replace(/\/+$/, '');
  }

  for (const [name, value] of Object.entries(headers)) {
    if (name === 'authorization') {
      const m = /^bearer\s+(.+)$/i.exec(value);
      result.apiKey = m ? m[1] : value;
    } else if (name !== 'content-type' && name !== 'accept' && name !== 'content-length') {
      result.headers[name] = value;
    }
  }

  let json: Record<string, unknown> | null = null;
  if (body) {
    try {
      json = JSON.parse(body) as Record<string, unknown>;
    } catch {
      result.warnings.push('Request body is not JSON; model not read from it.');
    }
  }
  const model = (json?.model as string | undefined) ?? formFields.model;
  if (role && model) result.models[role] = model;
  if (role === 'tts' && typeof json?.voice === 'string') result.voice = json.voice;
}
