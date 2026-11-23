// A tiny "fire once, remember it" event helper. The loader fires hero:play; if a listener registers after the
// event already fired (fast second visit), it still runs immediately.
declare global {
  interface Window {
    __fired?: Record<string, boolean>;
  }
}

export function fire(name: string) {
  window.__fired = { ...(window.__fired ?? {}), [name]: true };
  window.dispatchEvent(new Event(name));
}

export function onceFired(name: string, fn: () => void) {
  if (window.__fired?.[name]) {
    fn();
    return () => {};
  }
  const h = () => fn();
  window.addEventListener(name, h, { once: true });
  return () => window.removeEventListener(name, h);
}
