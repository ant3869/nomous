import * as React from "react";
import { ChevronDown } from "lucide-react";

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  children: React.ReactNode;
  onValueChange?: (value: string) => void;
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(({ className = "", children, onValueChange, ...props }, ref) => {
  const baseClasses =
    "flex h-10 w-full appearance-none rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 pr-8 text-sm text-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/70 focus-visible:border-emerald-500/50 disabled:cursor-not-allowed disabled:opacity-50 transition-colors";

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (onValueChange) {
      onValueChange(event.target.value);
    }
  };

  return (
    <div className="relative">
      <select className={`${baseClasses} ${className}`} ref={ref} onChange={handleChange} {...props}>
        {children}
      </select>
      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 pointer-events-none" />
    </div>
  );
});

Select.displayName = "Select";

// For compatibility with shadcn/ui patterns
const SelectTrigger = React.forwardRef<
  HTMLButtonElement,
  React.HTMLAttributes<HTMLButtonElement> & { children: React.ReactNode }
>(({ className = "", children, ...props }, ref) => {
  const baseClasses =
    "flex h-10 w-full items-center justify-between rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/70 disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <button type="button" role="combobox" className={`${baseClasses} ${className}`} ref={ref} {...props}>
      {children}
    </button>
  );
});

SelectTrigger.displayName = "SelectTrigger";

const SelectValue = ({ placeholder }: { placeholder?: string }) => {
  return <span className="text-zinc-400">{placeholder}</span>;
};

const SelectContent = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => {
  const baseClasses = "bg-zinc-900 border border-zinc-800 rounded-md p-1";
  return <div className={`${baseClasses} ${className}`}>{children}</div>;
};

const SelectItem = ({ value, children }: { value: string; children: React.ReactNode }) => {
  return (
    <option value={value} className="bg-zinc-900 text-zinc-100 hover:bg-zinc-800 px-2 py-1.5 rounded cursor-pointer">
      {children}
    </option>
  );
};

export { Select, SelectTrigger, SelectValue, SelectContent, SelectItem };
