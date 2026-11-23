// A real 3D plane drawn with CSS perspective: no WebGL (guide F5).
export function Horizon({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden className={"horizon " + className}>
      <div className="horizon-plane" />
    </div>
  );
}
