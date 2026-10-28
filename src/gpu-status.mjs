const MiB=1024**2;
function models(value){
  const items=Array.isArray(value)?value:value?.running??value?.models;
  if(!Array.isArray(items))throw Error('Invalid llama-swap running status');
  if(items.some(v=>typeof v!=='string'&&(!v||typeof(v.model??v.name)!=='string')))throw Error('Invalid running model entry');
  return items.length;
}
const unknown=blocked=>({phase:'unknown',blocked,llm:'unknown',warning:blocked?'GPU handoff status unavailable. Sending is paused until the active GPU handoff is verified.':'GPU status unavailable. No active GPU handoff is known; ordinary chat remains available.'});
export function createGpuStatus({comfyBaseUrl='http://127.0.0.1:8188',llamaSwapBaseUrl='http://127.0.0.1:11435',fetchImpl=globalThis.fetch}={}){
  const tasks=new Map();
  function update(value){
    if(!value?.prompt_id)return;
    if(['ready_for_review','submission_failed'].includes(value.phase))tasks.delete(value.prompt_id);
    else tasks.set(value.prompt_id,{phase:value.phase,at:Date.now(),terminal:value.execution_status==='completed'||value.execution_status==='failed'||value.phase==='cancelled'||value.phase==='unmonitored'});
    if(value.execution_status==='failed')lastWarning='Video generation failed. Check the job error before retrying.';
  }
  let lastWarning,knownGpuRiskAt=0;
  async function sample(){
    try{
      const get=async(base,path)=>{const r=await fetchImpl(new URL(path,base),{redirect:'error',signal:AbortSignal.timeout(4000)});if(!r.ok)throw Error('Service unavailable');return r.json();};
      const evidence=await Promise.allSettled([get(comfyBaseUrl,'/queue'),get(llamaSwapBaseUrl,'/running'),get(comfyBaseUrl,'/system_stats')]);
      const queue=evidence[0].value;
      if(Array.isArray(queue?.queue_running)&&Array.isArray(queue?.queue_pending)&&(queue.queue_running.length||queue.queue_pending.length))knownGpuRiskAt=Date.now();
      if(evidence.some(value=>value.status==='rejected'))throw Error('Incomplete GPU evidence');
      const running=evidence[1].value,stats=evidence[2].value;
      const queueCounts=q=>{if(!Array.isArray(q?.queue_running)||!Array.isArray(q?.queue_pending))throw Error('Invalid queue');return {running:q.queue_running.length,queued:q.queue_pending.length};};
      const llm=models(running)?'loaded':'unloaded';
      const devices=stats?.devices?.filter(d=>d.type==='cuda');
      if(!devices?.length||devices.some(d=>![d.vram_total,d.vram_free,d.torch_vram_total,d.torch_vram_free].every(v=>Number.isFinite(v)&&v>=0)||d.vram_total===0||d.vram_free>d.vram_total||d.torch_vram_free>d.torch_vram_total||d.torch_vram_total>d.vram_total))throw Error('Invalid GPU evidence');
      // The allocator floor is not a leak. This install runs cudaMallocAsync, which keeps a
      // residual reservation after /free (measured 32 MiB on a fresh process, 96 MiB after a
      // render), so a 64 MiB ceiling sits right on that boundary and can never certify an
      // honestly idle ComfyUI. Tolerate the floor and keep the physical-headroom test, which is
      // what actually decides whether the swap LLM has room to load.
      const torchFloor=d=>Math.max(256*MiB,Math.round(d.vram_total*0.02));
      const cacheFree=devices.every(d=>d.torch_vram_total<=torchFloor(d));
      const physicalFree=cacheFree&&devices.every(d=>(d.vram_free-d.torch_vram_free)/d.vram_total>=0.9);
      const counts=queueCounts(queue),busy=counts.running+counts.queued;
      const generating=c=>{knownGpuRiskAt=Date.now();return {phase:'generating',blocked:true,llm,...c,...(llm==='loaded'?{warning:'LLM is loaded while ComfyUI is running. Sending is paused to avoid a VRAM conflict.'}:{})};};
      if(busy)return generating(counts);
      // Cancelling a monitor is not release evidence. But a monitor that has stopped publishing
      // with a known terminal outcome must not hold the chat paused forever: verified release can
      // be unachievable while the swap LLM legitimately holds the GPU, and one stuck entry there
      // blocks sending until the whole harness is restarted. Sweep stale entries only when fresh
      // evidence says the ComfyUI queue is idle, then let the live GPU checks below decide.
      const sampleAt=Date.now();
      for(const [id,t] of [...tasks]){
        const cancelled=['cancelled','unmonitored'].includes(t.phase);
        const age=sampleAt-(t.at??sampleAt);
        // Hard ceiling: this sweep only runs while the ComfyUI queue is idle, so an entry with no
        // monitor refresh for 10 minutes describes a dead monitor, not a live render (a live render
        // keeps the queue busy and returns before this point). Expired entries stop pausing sending,
        // and the expiry is reported instead of being presented as release evidence.
        if(age>600_000){lastWarning='Handoff state expired after 10 minutes with the ComfyUI queue idle and no monitor update; sending resumed. This is not release evidence: check live GPU evidence before queueing another generation.';tasks.delete(id);continue;}
        const stale=(t.terminal||cancelled)&&age>60_000;
        if(stale||(cancelled&&physicalFree))tasks.delete(id);
      }
      const active=[...tasks.values()];
      if(active.length)return {phase:active.some(t=>['queued','unloading_llm','generating'].includes(t.phase))?'generating':'releasing_vram',blocked:true,llm,...(active.some(t=>['monitor_error','waiting_for_history','unmonitored'].includes(t.phase))?{warning:'GPU handoff is not verified. Check the job monitor; do not resubmit the video.'}:{})};
      if(!cacheFree||(llm==='unloaded'&&!physicalFree)){knownGpuRiskAt=Date.now();return {phase:'releasing_vram',blocked:true,llm,warning:'ComfyUI memory is not released yet. Sending stays paused until GPU release is verified.'};}
      // Statistics may be slow. Do not publish old idle evidence after a new prompt.
      const finalQueue=queueCounts(await get(comfyBaseUrl,'/queue'));
      if(finalQueue.running+finalQueue.queued)return generating(finalQueue);
      if(tasks.size)return {phase:'releasing_vram',blocked:true,llm};
      knownGpuRiskAt=0;
      return {phase:'ready',blocked:false,llm,...(lastWarning?{warning:lastWarning}:{})};
    }catch{return unknown(tasks.size>0||Date.now()-knownGpuRiskAt<90_000);}
  }
  async function recover(){
    if(tasks.size)throw Error('A managed ComfyUI handoff is still active');
    const get=async(base,path)=>{
      const response=await fetchImpl(new URL(path,base),{redirect:'error',signal:AbortSignal.timeout(4000)});
      if(!response.ok)throw Error(`${path} unavailable`);
      return response.json();
    };
    const checkIdle=async()=>{
      const [queue,running]=await Promise.all([get(comfyBaseUrl,'/queue'),get(llamaSwapBaseUrl,'/running')]);
      if(!Array.isArray(queue?.queue_running)||!Array.isArray(queue?.queue_pending)||queue.queue_running.length||queue.queue_pending.length)throw Error('ComfyUI queue is busy or unavailable');
      if(models(running))throw Error('LLM model is loaded');
    };
    await checkIdle();
    const current=await sample();
    if(current.phase==='ready')return current;
    if(current.phase!=='releasing_vram'||current.llm!=='unloaded')throw Error('GPU release cannot be verified now');
    await checkIdle();
    const response=await fetchImpl(new URL('/free',comfyBaseUrl),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({unload_models:true,free_memory:true}),redirect:'error',signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw Error(`ComfyUI memory release failed (${response.status})`);
    return sample();
  }
  // Every request starts its own evidence sample, including an explicit Send.
  return {update,snapshot:sample,recover};
}
export function createGpuStatusHandler({status,requestRejection}){
  return async(req,res)=>{
    const rejection=requestRejection(req);if(rejection!==undefined){res.writeHead(rejection);res.end();return;}
    if(req.method!=='GET'){res.writeHead(405,{Allow:'GET'});res.end();return;}
    const value=await status.snapshot();if(res.destroyed)return;
    res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'});res.end(JSON.stringify(value));
  };
}
export function registerGpuStatusRoute(ctx,status){return ctx.webServer.register({kind:'exact',path:'/director-gpu-status',handler:createGpuStatusHandler({status,requestRejection:req=>ctx.connection.requestRejection(req)})});}
export function createGpuReleaseHandler({status,requestRejection}){
  return async(req,res)=>{
    const rejection=requestRejection(req);if(rejection!==undefined){res.writeHead(rejection);res.end();return;}
    if(req.method!=='POST'){res.writeHead(405,{Allow:'POST'});res.end();return;}
    try{
      const value=await status.recover();if(res.destroyed)return;
      res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'});res.end(JSON.stringify(value));
    }catch(error){
      if(res.destroyed)return;
      res.writeHead(409,{'Content-Type':'application/json','Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'});res.end(JSON.stringify({error:error.message}));
    }
  };
}
export function registerGpuReleaseRoute(ctx,status){return ctx.webServer.register({kind:'exact',path:'/director-gpu-release',handler:createGpuReleaseHandler({status,requestRejection:req=>ctx.connection.requestRejection(req)})});}
