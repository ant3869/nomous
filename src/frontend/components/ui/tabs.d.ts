import { HTMLAttributes, ReactNode } from "react";

export interface TabsProps extends HTMLAttributes<HTMLDivElement> {
  defaultValue: string;
  value?: string;
  onValueChange?: (value: string) => void;
  children: ReactNode;
}

export interface TabsListProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export interface TabsTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  value: string;
  children: ReactNode;
}

export interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
  value: string;
  children: ReactNode;
}

export declare const Tabs: (props: TabsProps) => JSX.Element;
export declare const TabsList: (props: TabsListProps) => JSX.Element;
export declare const TabsTrigger: (props: TabsTriggerProps) => JSX.Element;
export declare const TabsContent: (props: TabsContentProps) => JSX.Element;