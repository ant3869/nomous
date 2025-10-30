export function Switch({checked, onCheckedChange}:{checked:boolean; onCheckedChange:(v:boolean)=>void}){
  return (
    <button onClick={()=>onCheckedChange(!checked)} className={`w-10 h-6 rounded-full transition ${checked?'bg-emerald-600/90':'bg-zinc-700'}`}>
      <div className={`h-5 w-5 bg-white rounded-full translate-y-0.5 transition ${checked?'translate-x-5':'translate-x-0.5'}`}/>
    </button>
  )
}
