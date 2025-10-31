import React, { useCallback, useRef } from "react";
import { FolderOpen } from "lucide-react";
import { Button } from "./ui/button";

type FilePathInputProps = {
  value: string;
  onChange: (value: string) => void;
  onCommit?: (value: string) => void;
  placeholder?: string;
  className?: string;
  browseLabel?: string;
  allowDirectories?: boolean;
  accept?: string[];
  disabled?: boolean;
};

export function FilePathInput({
  value,
  onChange,
  onCommit,
  placeholder,
  className = "",
  browseLabel = "Browse",
  allowDirectories = false,
  accept,
  disabled = false,
}: FilePathInputProps) {
  const hiddenFileInput = useRef<HTMLInputElement | null>(null);

  const triggerCommit = useCallback(
    (next: string) => {
      if (onCommit) {
        onCommit(next);
      }
    },
    [onCommit]
  );

  const handleBrowse = useCallback(async () => {
    if (disabled) return;

    try {
      if (typeof window !== "undefined") {
        if (!allowDirectories && "showOpenFilePicker" in window) {
          const opts: any = { multiple: false };
          if (accept && accept.length > 0) {
            opts.types = [
              {
                description: "Model files",
                accept: {
                  "application/octet-stream": accept,
                },
              },
            ];
          }
          const handles = await (window as any).showOpenFilePicker(opts);
          if (handles && handles.length > 0) {
            const handle = handles[0];
            const file = await handle.getFile();
            const syntheticPath =
              (handle as any).fullPath ||
              (file as any).path ||
              file.webkitRelativePath ||
              file.name;
            onChange(syntheticPath);
            triggerCommit(syntheticPath);
            return;
          }
        }
        if (allowDirectories && "showDirectoryPicker" in window) {
          const dirHandle = await (window as any).showDirectoryPicker();
          if (dirHandle) {
            const syntheticPath = (dirHandle as any).fullPath || dirHandle.name;
            onChange(syntheticPath);
            triggerCommit(syntheticPath);
            return;
          }
        }
      }
    } catch (err: any) {
      if (err?.name === "AbortError") {
        return;
      }
      // Fall back to hidden file input below.
    }

    hiddenFileInput.current?.click();
  }, [accept, allowDirectories, disabled, onChange, triggerCommit]);

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }
      const derivedPath =
        (file as any).path || file.webkitRelativePath || file.name;
      onChange(derivedPath);
      triggerCommit(derivedPath);
      event.target.value = "";
    },
    [onChange, triggerCommit]
  );

  const handleBlur = useCallback(() => {
    triggerCommit(value);
  }, [triggerCommit, value]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "Enter") {
        event.preventDefault();
        triggerCommit(value);
      }
    },
    [triggerCommit, value]
  );

  return (
    <div className={`flex gap-2 ${className}`}>
      <input
        type="text"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="flex-1 rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus:border-emerald-500/80 focus:outline-none focus:ring-0 disabled:cursor-not-allowed disabled:opacity-50"
      />
      <input
        ref={hiddenFileInput}
        type="file"
        className="hidden"
        onChange={handleFileChange}
        {...(allowDirectories
          ? ({ webkitdirectory: "true", directory: "true" } as any)
          : {})}
        {...(accept && accept.length > 0
          ? ({ accept: accept.join(",") } as React.InputHTMLAttributes<HTMLInputElement>)
          : {})}
      />
      <Button
        type="button"
        variant="secondary"
        onClick={handleBrowse}
        disabled={disabled}
        className="shrink-0 px-3 py-2"
      >
        <FolderOpen className="h-4 w-4" />
        <span className="sr-only md:not-sr-only md:inline">{browseLabel}</span>
      </Button>
    </div>
  );
}

export default FilePathInput;
