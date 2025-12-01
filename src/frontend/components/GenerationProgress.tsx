import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

interface GenerationProgressProps {
  text: string;
  tokens: number;
}

export function GenerationProgress({ text, tokens }: GenerationProgressProps) {
  // Estimate progress based on token count (rough heuristic)
  // Typical response: 50-500 tokens, we'll map this to a progress bar
  const estimatedMax = 200; // Assume typical response is ~200 tokens
  const progress = Math.min(100, (tokens / estimatedMax) * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex flex-col items-center gap-2 py-3"
    >
      {/* Animated spinner */}
      <div className="flex items-center gap-2 text-sky-400">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-sm font-medium">Generating Response</span>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-xs">
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
          <motion.div
            className="h-full bg-gradient-to-r from-sky-500 to-emerald-500"
            initial={{ width: "0%" }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Token count label */}
      <div className="text-xs text-zinc-400">
        {tokens} tokens • {text}
      </div>
    </motion.div>
  );
}
