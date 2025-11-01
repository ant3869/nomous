import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from "react";

type TabsContextValue = {
  value: string;
  setValue: (next: string) => void;
};

const TabsContext = createContext<TabsContextValue | undefined>(undefined);

function useTabsContext(component: string): TabsContextValue {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error(`${component} must be used within <Tabs>`);
  }
  return context;
}

const cx = (...classes: Array<string | false | null | undefined>) =>
  classes.filter(Boolean).join(" ");

export interface TabsProps extends HTMLAttributes<HTMLDivElement> {
  defaultValue: string;
  value?: string;
  onValueChange?: (value: string) => void;
  children: ReactNode;
}

export const Tabs = ({
  defaultValue,
  value: controlledValue,
  onValueChange,
  children,
  className,
  ...props
}: TabsProps) => {
  const [uncontrolledValue, setUncontrolledValue] = useState(defaultValue);

  const value = controlledValue ?? uncontrolledValue;

  const setValue = useCallback(
    (next: string) => {
      setUncontrolledValue(next);
      onValueChange?.(next);
    },
    [onValueChange]
  );

  const contextValue = useMemo<TabsContextValue>(
    () => ({ value, setValue }),
    [value, setValue]
  );

  return (
    <TabsContext.Provider value={contextValue}>
      <div className={cx("inline-flex w-full flex-col gap-2", className)} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
};

export interface TabsListProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const TabsList = ({ className, children, ...props }: TabsListProps) => (
  <div
    role="tablist"
    className={cx("inline-flex flex-wrap items-center gap-2", className)}
    {...props}
  >
    {children}
  </div>
);

export interface TabsTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  value: string;
  children: ReactNode;
}

export const TabsTrigger = ({
  value,
  children,
  className,
  ...props
}: TabsTriggerProps) => {
  const { value: activeValue, setValue } = useTabsContext("TabsTrigger");
  const isActive = activeValue === value;

  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      data-state={isActive ? "active" : "inactive"}
      className={cx(
        "rounded-md border border-transparent px-3 py-1 text-sm font-medium transition-colors",
        isActive
          ? "bg-zinc-800 text-white shadow"
          : "bg-zinc-900/70 text-zinc-200 hover:border-zinc-700 hover:bg-zinc-900",
        className
      )}
      onClick={() => setValue(value)}
      {...props}
    >
      {children}
    </button>
  );
};

export interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
  value: string;
  children: ReactNode;
}

export const TabsContent = ({
  value,
  children,
  className,
  ...props
}: TabsContentProps) => {
  const { value: activeValue } = useTabsContext("TabsContent");

  if (activeValue !== value) {
    return null;
  }

  return (
    <div
      role="tabpanel"
      data-state="active"
      className={cx("w-full", className)}
      {...props}
    >
      {children}
    </div>
  );
};

