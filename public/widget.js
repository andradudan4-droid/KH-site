
(function(){
  var base = new URL(document.currentScript.src).origin;
  var bubble = document.createElement('button');
  bubble.id = 'kh-chat-bubble';
  bubble.innerHTML = 'Chat';
  bubble.setAttribute('aria-label','Open quote assistant');
  bubble.style.cssText='position:fixed;right:22px;bottom:22px;z-index:999999;border:1px solid rgba(242,245,248,.28);border-radius:999px;background:linear-gradient(135deg,#050506,#444b55 58%,#f2f5f8);color:white;font-weight:900;padding:15px 18px;box-shadow:0 16px 38px rgba(0,0,0,.46),0 0 22px rgba(242,245,248,.18);cursor:pointer';
  var frame = document.createElement('iframe');
  frame.src = base + '/widget-frame';
  function size(){ frame.style.cssText = window.innerWidth <= 640 ? 'position:fixed;inset:0;width:100vw;height:100dvh;border:0;z-index:999999;display:none;background:white' : 'position:fixed;right:22px;bottom:84px;width:410px;height:610px;border:0;border-radius:18px;box-shadow:0 18px 60px rgba(0,0,0,.45);z-index:999999;display:none;background:white'; }
  size(); window.addEventListener('resize',size);
  bubble.onclick=function(){ frame.style.display='block'; bubble.style.display=window.innerWidth<=640?'none':'block'; document.body.style.overflow=window.innerWidth<=640?'hidden':''; };
  window.addEventListener('message',function(e){ if(e.data==='close-au-chat'){ frame.style.display='none'; bubble.style.display='block'; document.body.style.overflow=''; }});
  document.body.appendChild(bubble); document.body.appendChild(frame);
})();
