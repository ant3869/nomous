import React from 'react'
export function Card(props: React.HTMLAttributes<HTMLDivElement>) {
  return <div {...props} className={`card ${props.className||''}`} />
}
export function CardContent(props: React.HTMLAttributes<HTMLDivElement>) {
  return <div {...props} className={`p-4 ${props.className||''}`} />
}
export function CardHeader(props: React.HTMLAttributes<HTMLDivElement>) {
  return <div {...props} className={`px-4 pt-4 pb-2 ${props.className||''}`} />
}
export function CardTitle(props: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 {...props} className={`text-lg font-semibold ${props.className||''}`} />
}
