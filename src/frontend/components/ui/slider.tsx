import React from 'react'
export function Slider({defaultValue=[0], min=0, max=100, step=1, onValueChange}:{defaultValue?:number[];min?:number;max?:number;step?:number;onValueChange?:(v:number[])=>void}){
  const [v,setV]=React.useState(defaultValue[0])
  return (
    <input type="range" min={min} max={max} step={step} defaultValue={v}
      onChange={(e)=>{const n=Number(e.target.value);setV(n);onValueChange&&onValueChange([n])}}
      className="w-full accent-emerald-500"/>
  )
}
