import * as React from "react";

export interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  children: React.ReactNode;
  onValueChange?: (value: string) => void;
}

export const Select: React.ForwardRefExoticComponent<SelectProps & React.RefAttributes<HTMLSelectElement>>;
export const SelectTrigger: React.ForwardRefExoticComponent<React.HTMLAttributes<HTMLButtonElement> & { children: React.ReactNode } & React.RefAttributes<HTMLButtonElement>>;
export const SelectValue: React.FC<{ placeholder?: string }>;
export const SelectContent: React.FC<{ children: React.ReactNode; className?: string }>;
export const SelectItem: React.FC<{ value: string; children: React.ReactNode }>;
