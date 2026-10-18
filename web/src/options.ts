/** A copy of `o` without the keys whose value is `undefined`, so `{ ...defaults, ...defined(o) }` never lets an unset field erase a default. */
export function defined<T extends object>(o: T | undefined): Partial<T> {
  const out: Partial<T> = {};
  if (o) for (const k of Object.keys(o) as (keyof T)[]) if (o[k] !== undefined) out[k] = o[k];
  return out;
}
