import { FC, HTMLAttributes } from 'react';

export interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  value: number;
  max?: number;
  className?: string;
}

export declare const Progress: FC<ProgressProps>;