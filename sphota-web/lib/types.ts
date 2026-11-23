export type Box = { x: number; y: number; w: number; h: number; kind: "text" | "face"; label: string };
export type Mode = "exact" | "balanced" | "small" | "tiny";

export type CompressResult = {
  job_id: string;
  bytes: number;
  original_bytes: number;
  bpp: number;
  mode: Mode;
  recommended_mode: Mode | null;
  warnings?: string[];
  boxes: Box[];
  timings: Record<string, number>;
  seed_url: string;
};

export type Flag = { id: string; name: string; reason: string; score: number; box: Box; icon: string };

export type DoneEvent = {
  stage: "done";
  image_url: string;
  coarse_url: string;
  uncertainty_url: string;
  consistency: number;
  flagged: Flag[];
};

export type JobEvent =
  | { stage: "detect" | "recommend" | "encode" | "entropy" | "pack"; status: "running" | "done"; ms?: number; delay?: number }
  | { stage: "denoise"; step: number; total: number; preview_url: string; delay?: number }
  | (DoneEvent & { delay?: number })
  | { stage: "error"; message: string; delay?: number };

export type DecoderResult = { url: string; lpips: number | null; verdict: string; kept: boolean };
export type CompareResult = {
  jpeg: DecoderResult;
  coarse: DecoderResult;
  plain: DecoderResult;
  sphota: DecoderResult;
  metrics: {
    tfr: { plain: number | null; sphota: number | null };
    fir: { plain: number | null; sphota: number | null };
    consistency: { plain: number; sphota: number };
    decode_s: { plain: number; sphota: number };
  };
};

export type TreeNode = { id: string; label: string; depth: number; col: number; taken: boolean; leaf: boolean; chosen?: boolean };
export type TreeEdge = { from: string; to: string; label?: string; taken: boolean };
export type Rule = { antecedents: string[]; consequent: string; support: number; confidence: number; lift: number; fired: boolean; note?: string };
export type ExplainResult = {
  tree: { nodes: TreeNode[]; edges: TreeEdge[] };
  rules: Rule[];
  neighbours: { thumb_url: string; best_mode: Mode; name: string }[];
  predicted: { kb: number; kb_std: number; lpips: number };
  actual: { kb: number; lpips: number };
};
