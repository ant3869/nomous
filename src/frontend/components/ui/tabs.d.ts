import { HTMLAttributes } from 'react';

export interface TabsProps extends HTMLAttributes<HTMLDivElement> {
  defaultValue: string;
  className?: string;
}

export interface TabsListProps extends HTMLAttributes<HTMLDivElement> {
  className?: string;
}

export interface TabsTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  value: string;
  className?: string;
}

export interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
  value: string;
  className?: string;
}

export declare const Tabs: React.FC<TabsProps>;
export declare const TabsList: React.FC<TabsListProps>;
export declare const TabsTrigger: React.FC<TabsTriggerProps>;
export declare const TabsContent: React.FC<TabsContentProps>;