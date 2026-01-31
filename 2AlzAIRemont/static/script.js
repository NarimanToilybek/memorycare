const mriForm=document.getElementById('mriForm');
if(mriForm){
  mriForm.addEventListener('submit',async e=>{
    if(!mriForm.checkValidity()){e.preventDefault();mriForm.classList.add('was-validated');return;}
    e.preventDefault();
    const file=mriForm.querySelector('#mriFile').files[0];
    const resArea=document.getElementById('resultArea');
    resArea.className='alert alert-info mt-4';resArea.textContent='⏳ Анализ…';
    const fd=new FormData();fd.append('file',file);
    try{const r=await fetch('/api/analyze',{method:'POST',body:fd});if(!r.ok)throw'';const d=await r.json();
      resArea.className='alert alert-success mt-4';resArea.innerHTML=`✅ <strong>${d.stage}</strong><br>${d.description}`;
    }catch{resArea.className='alert alert-danger mt-4';resArea.textContent='Ошибка сервера.';}
  });
}