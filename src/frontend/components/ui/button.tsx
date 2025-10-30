import React from 'react'
type V = 'primary' | 'secondary' | 'danger'
export function Button({variant='primary', className='', ...p}:{variant?:V}&React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const style = variant==='primary'?'btn btn-primary':variant==='danger'?'btn btn-danger':'btn btn-secondary'
  return <button {...p} className={`${style} ${className}`} />
}
