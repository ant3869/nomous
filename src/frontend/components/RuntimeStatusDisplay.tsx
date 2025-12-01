import React, { useMemo, useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { NomousStatus } from "../App";

interface RuntimeStatusDisplayProps {
  status: NomousStatus;
  statusDetail?: string;
}

// Status configuration with personalities and emotions
const statusConfig: Record<NomousStatus, {
  label: string;
  sublabel: string;
  emoji: string;
  primaryColor: string;
  secondaryColor: string;
  glowColor: string;
  pulseSpeed: number;
  personality: string[];
  animation: "breathe" | "pulse" | "wave" | "glow" | "spark";
}> = {
  idle: {
    label: "Idle",
    sublabel: "Ready",
    emoji: "💭",
    primaryColor: "#a1a1aa",
    secondaryColor: "#52525b",
    glowColor: "rgba(161, 161, 170, 0.4)",
    pulseSpeed: 4,
    personality: ["Listening...", "At your service", "Standing by", "Awaiting input", "Present and ready"],
    animation: "breathe"
  },
  thinking: {
    label: "Thinking",
    sublabel: "Processing...",
    emoji: "🧠",
    primaryColor: "#a855f7",
    secondaryColor: "#7c3aed",
    glowColor: "rgba(168, 85, 247, 0.5)",
    pulseSpeed: 0.8,
    personality: ["Contemplating...", "Processing thoughts...", "Analyzing...", "Considering...", "Deep in thought..."],
    animation: "pulse"
  },
  speaking: {
    label: "Speaking",
    sublabel: "Communicating",
    emoji: "🗣️",
    primaryColor: "#10b981",
    secondaryColor: "#059669",
    glowColor: "rgba(16, 185, 129, 0.5)",
    pulseSpeed: 0.6,
    personality: ["Expressing...", "Sharing thoughts...", "Communicating...", "Speaking my mind...", "Voicing..."],
    animation: "wave"
  },
  noticing: {
    label: "Noticing",
    sublabel: "Observing",
    emoji: "👁️",
    primaryColor: "#f59e0b",
    secondaryColor: "#d97706",
    glowColor: "rgba(245, 158, 11, 0.5)",
    pulseSpeed: 1.2,
    personality: ["Observing...", "Something caught my eye...", "Perceiving...", "Watching...", "Noticing changes..."],
    animation: "spark"
  },
  learning: {
    label: "Learning",
    sublabel: "Evolving",
    emoji: "✨",
    primaryColor: "#06b6d4",
    secondaryColor: "#0891b2",
    glowColor: "rgba(6, 182, 212, 0.5)",
    pulseSpeed: 1.0,
    personality: ["Growing...", "Adapting...", "Absorbing knowledge...", "Evolving...", "Learning..."],
    animation: "glow"
  },
  error: {
    label: "Error",
    sublabel: "Attention needed",
    emoji: "⚠️",
    primaryColor: "#ef4444",
    secondaryColor: "#dc2626",
    glowColor: "rgba(239, 68, 68, 0.5)",
    pulseSpeed: 0.5,
    personality: ["Something went wrong...", "Need assistance...", "Encountering issues...", "Requires attention..."],
    animation: "pulse"
  }
};

// Particles for visual flair
const Particle: React.FC<{ delay: number; color: string }> = ({ delay, color }) => (
  <motion.div
    className="absolute w-1 h-1 rounded-full"
    style={{ backgroundColor: color }}
    initial={{ opacity: 0, scale: 0, x: 0, y: 0 }}
    animate={{
      opacity: [0, 1, 0],
      scale: [0, 1.5, 0],
      x: [0, (Math.random() - 0.5) * 60],
      y: [0, (Math.random() - 0.5) * 60],
    }}
    transition={{
      duration: 2,
      delay,
      repeat: Infinity,
      repeatDelay: Math.random() * 2,
    }}
  />
);

// Audio visualizer bars for speaking state
const AudioBar: React.FC<{ index: number; color: string }> = ({ index, color }) => (
  <motion.div
    className="w-1 rounded-full"
    style={{ backgroundColor: color }}
    animate={{
      height: [4, 12 + Math.random() * 16, 4],
    }}
    transition={{
      duration: 0.4 + Math.random() * 0.3,
      delay: index * 0.1,
      repeat: Infinity,
      repeatType: "reverse",
    }}
  />
);

// Neural network animation for thinking state
const NeuralNode: React.FC<{ index: number; color: string }> = ({ index, color }) => {
  const angle = (index / 8) * Math.PI * 2;
  const radius = 20;
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  
  return (
    <motion.div
      className="absolute w-2 h-2 rounded-full"
      style={{
        backgroundColor: color,
        left: `calc(50% + ${x}px)`,
        top: `calc(50% + ${y}px)`,
      }}
      animate={{
        scale: [1, 1.5, 1],
        opacity: [0.4, 1, 0.4],
      }}
      transition={{
        duration: 1.5,
        delay: index * 0.15,
        repeat: Infinity,
      }}
    />
  );
};

// Main component
export const RuntimeStatusDisplay: React.FC<RuntimeStatusDisplayProps> = ({
  status,
  statusDetail,
}) => {
  const config = statusConfig[status];
  const [personalityIndex, setPersonalityIndex] = useState(0);
  const [displayText, setDisplayText] = useState(config.personality[0]);
  const prevStatusRef = useRef(status);
  
  // Rotate personality messages
  useEffect(() => {
    const interval = setInterval(() => {
      setPersonalityIndex((prev) => (prev + 1) % config.personality.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [config.personality.length]);
  
  // Animate text typing effect
  useEffect(() => {
    const target = statusDetail || config.personality[personalityIndex];
    let index = 0;
    setDisplayText("");
    
    const typeInterval = setInterval(() => {
      if (index <= target.length) {
        setDisplayText(target.substring(0, index));
        index++;
      } else {
        clearInterval(typeInterval);
      }
    }, 30);
    
    return () => clearInterval(typeInterval);
  }, [personalityIndex, statusDetail, config.personality]);
  
  // Reset personality index on status change
  useEffect(() => {
    if (prevStatusRef.current !== status) {
      setPersonalityIndex(0);
      prevStatusRef.current = status;
    }
  }, [status]);
  
  // Memoize particles to prevent re-creation
  const particles = useMemo(() => {
    if (status !== "thinking" && status !== "learning") return null;
    return Array.from({ length: 8 }).map((_, i) => (
      <Particle key={i} delay={i * 0.25} color={config.primaryColor} />
    ));
  }, [status, config.primaryColor]);
  
  const neuralNodes = useMemo(() => {
    if (status !== "thinking") return null;
    return Array.from({ length: 8 }).map((_, i) => (
      <NeuralNode key={i} index={i} color={config.primaryColor} />
    ));
  }, [status, config.primaryColor]);
  
  const audioBars = useMemo(() => {
    if (status !== "speaking") return null;
    return Array.from({ length: 5 }).map((_, i) => (
      <AudioBar key={i} index={i} color={config.primaryColor} />
    ));
  }, [status, config.primaryColor]);

  return (
    <div className="relative overflow-hidden rounded-2xl border border-zinc-800/70 bg-zinc-950/70 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      {/* Animated gradient background */}
      <motion.div
        className="absolute inset-0 opacity-30"
        animate={{
          background: [
            `radial-gradient(circle at 20% 50%, ${config.glowColor} 0%, transparent 50%)`,
            `radial-gradient(circle at 80% 50%, ${config.glowColor} 0%, transparent 50%)`,
            `radial-gradient(circle at 50% 20%, ${config.glowColor} 0%, transparent 50%)`,
            `radial-gradient(circle at 50% 80%, ${config.glowColor} 0%, transparent 50%)`,
            `radial-gradient(circle at 20% 50%, ${config.glowColor} 0%, transparent 50%)`,
          ],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      
      {/* Scan line effect */}
      <motion.div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `linear-gradient(transparent 50%, rgba(0,0,0,0.1) 50%)`,
          backgroundSize: "100% 4px",
        }}
        animate={{ opacity: [0.3, 0.5, 0.3] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
      
      <div className="relative px-5 py-4">
        <div className="flex items-center gap-4">
          {/* Status orb with animations */}
          <div className="relative flex items-center justify-center">
            {/* Outer glow ring */}
            <motion.div
              className="absolute rounded-full"
              style={{
                width: 56,
                height: 56,
                background: `radial-gradient(circle, ${config.glowColor} 0%, transparent 70%)`,
              }}
              animate={{
                scale: [1, 1.3, 1],
                opacity: [0.5, 0.8, 0.5],
              }}
              transition={{
                duration: config.pulseSpeed,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
            
            {/* Middle ring */}
            <motion.div
              className="absolute rounded-full border-2"
              style={{
                width: 44,
                height: 44,
                borderColor: config.primaryColor,
              }}
              animate={{
                scale: [1, 1.1, 1],
                opacity: [0.3, 0.6, 0.3],
                rotate: [0, 180, 360],
              }}
              transition={{
                duration: config.pulseSpeed * 2,
                repeat: Infinity,
                ease: "linear",
              }}
            />
            
            {/* Core orb */}
            <motion.div
              className="relative flex items-center justify-center rounded-full"
              style={{
                width: 32,
                height: 32,
                background: `linear-gradient(135deg, ${config.primaryColor} 0%, ${config.secondaryColor} 100%)`,
                boxShadow: `0 0 20px ${config.glowColor}, inset 0 0 10px rgba(255,255,255,0.2)`,
              }}
              animate={
                config.animation === "breathe"
                  ? { scale: [1, 1.05, 1] }
                  : config.animation === "pulse"
                    ? { scale: [1, 1.15, 1] }
                    : config.animation === "wave"
                      ? { scale: [1, 1.1, 1, 1.05, 1] }
                      : config.animation === "spark"
                        ? { scale: [1, 1.2, 0.95, 1.1, 1] }
                        : { scale: [1, 1.08, 1] }
              }
              transition={{
                duration: config.pulseSpeed,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            >
              {/* Inner highlight */}
              <div
                className="absolute top-1 left-1 w-3 h-3 rounded-full opacity-40"
                style={{
                  background: "radial-gradient(circle, white 0%, transparent 70%)",
                }}
              />
              
              {/* Neural nodes for thinking */}
              {neuralNodes}
              
              {/* Particles */}
              {particles}
            </motion.div>
            
            {/* Audio visualizer for speaking */}
            {status === "speaking" && (
              <div className="absolute -bottom-2 flex items-end gap-0.5 h-6">
                {audioBars}
              </div>
            )}
          </div>
          
          {/* Status text */}
          <div className="flex-1 min-w-0">
            <div className="text-[10px] uppercase tracking-[0.4em] text-zinc-500 mb-1">
              Runtime Status
            </div>
            
            <AnimatePresence mode="wait">
              <motion.div
                key={status}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="flex items-center gap-2"
              >
                <span
                  className="text-xl font-semibold"
                  style={{ color: config.primaryColor }}
                >
                  {config.label}
                </span>
                <motion.span
                  className="text-lg"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                >
                  {config.emoji}
                </motion.span>
              </motion.div>
            </AnimatePresence>
            
            {/* Animated personality text */}
            <div className="h-5 overflow-hidden">
              <motion.div
                className="text-sm text-zinc-400 truncate"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                {displayText}
                <motion.span
                  className="inline-block w-0.5 h-4 ml-0.5 bg-zinc-400"
                  animate={{ opacity: [1, 0, 1] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                />
              </motion.div>
            </div>
          </div>
          
          {/* Mood indicator */}
          <div className="hidden md:flex flex-col items-end gap-1">
            <div className="text-[9px] uppercase tracking-[0.3em] text-zinc-600">
              Mood
            </div>
            <div className="flex items-center gap-1">
              {[...Array(5)].map((_, i) => (
                <motion.div
                  key={i}
                  className="w-1.5 h-1.5 rounded-full"
                  style={{
                    backgroundColor:
                      status === "error"
                        ? i < 2 ? config.primaryColor : "rgb(63 63 70)"
                        : status === "idle"
                          ? i < 3 ? config.primaryColor : "rgb(63 63 70)"
                          : i < 4 ? config.primaryColor : "rgb(63 63 70)",
                  }}
                  animate={
                    i < (status === "error" ? 2 : status === "idle" ? 3 : 4)
                      ? { scale: [1, 1.3, 1] }
                      : {}
                  }
                  transition={{
                    duration: 1,
                    delay: i * 0.1,
                    repeat: Infinity,
                  }}
                />
              ))}
            </div>
          </div>
        </div>
        
        {/* Activity indicator line */}
        <motion.div
          className="absolute bottom-0 left-0 h-0.5"
          style={{ backgroundColor: config.primaryColor }}
          animate={{
            width: ["0%", "100%", "0%"],
            left: ["0%", "0%", "100%"],
          }}
          transition={{
            duration: config.pulseSpeed * 3,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>
    </div>
  );
};

export default RuntimeStatusDisplay;
