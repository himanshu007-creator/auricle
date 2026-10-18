// Tiny typed event emitter.
export class Emitter<Events extends Record<string, unknown>> {
  private listeners = new Map<keyof Events, Set<(payload: never) => void>>();

  on<K extends keyof Events>(name: K, fn: (payload: Events[K]) => void): () => void {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set());
    this.listeners.get(name)!.add(fn as (payload: never) => void);
    return () => this.listeners.get(name)?.delete(fn as (payload: never) => void);
  }

  emit<K extends keyof Events>(name: K, payload: Events[K]): void {
    this.listeners.get(name)?.forEach((fn) => (fn as (p: Events[K]) => void)(payload));
  }
}
