import { HTMLAttributes } from 'react';

export interface TabsListProps extends HTMLAttributes<HTMLDivElement> {
  className?: string;
}

export const TabsList: React.FC<TabsListProps> = ({ className, children, ...props }) => {
  return (
    <div className={className} {...props}>
      {children}
    </div>
  );
};