import React from 'react'
export function Tabs({defaultValue, children}:{defaultValue:string; children:any}){
  const [v,setV]=React.useState(defaultValue)
  return <TabsCtx.Provider value={{v,setV}}>{children}</TabsCtx.Provider>
}
const TabsCtx = React.createContext<{v:string,setV:(s:string)=>void}>({v:'',setV:()=>{}})
export function TabsList({children}:{children:any}){ return <div className="tabs">{children}</div> }
export function TabsTrigger({value, children}:{value:string;children:any}){
  const {v,setV}=React.useContext(TabsCtx)
  const active = v===value
  return <button className={`tab ${active?'active':''}`} onClick={()=>setV(value)}>{children}</button>
}
export function TabsContent({value, children, className=''}:{value:string;children:any;className?:string}){
  const {v}=React.useContext(TabsCtx)
  return v===value?<div className={className}>{children}</div>:null
}
