import React from 'react'
export function Card(props: React.HTMLAttributes<HTMLDivElement>) {
  return <div {...props} className={`card ${props.className||''}`} />
}
export function CardContent(props: React.HTMLAttributes<HTMLDivElement>) {
  return <div {...props} className={`p-4 ${props.className||''}`} />
}
