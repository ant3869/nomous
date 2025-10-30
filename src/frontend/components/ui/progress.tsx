export function Progress({value=0}:{value?:number}){
  return (
    <div className="h-2 w-full rounded bg-zinc-800 overflow-hidden">
      <div className="h-full bg-emerald-500 transition-all" style={{width:`${Math.min(100,Math.max(0,value))}%`}}/>
    </div>
  )
}
