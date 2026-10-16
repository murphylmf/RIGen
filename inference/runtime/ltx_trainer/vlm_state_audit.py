''
import json
import math


def reconcile(state, observation, *, fired, min_confidence=.55):
    s=dict(state)
    status=observation['status']
    if observation['confidence'] < min_confidence or status=='uncertain':
        return s
    if status=='alive':
        prior=s['hp']
        hp=100 if s['phase']=='reentry' else (prior if prior>0 else s['last_alive_hp'])
                                                                    
        if fired and observation['shot_result']=='hit' and hp>50:
            hp-=50
        s.update(hp=hp,last_alive_hp=hp,phase='active',corpse_blocks=0)
    elif status=='dead':
        count=s['corpse_blocks']+1 if s['phase']=='corpse' else 0
        s.update(hp=0,phase='despawn' if count>=2 else 'corpse',corpse_blocks=count)
    elif status=='absent':
                                                                         
                                                             
        if s['phase']=='despawn' and s['hp']==0:
            s['phase']='reentry'
    return s


def observe(observer, *, before, after, guidance, fired):
    import requests
    schema={'name':'enemy_video_state','strict':True,'schema':{
        'type':'object','properties':{
            'status':{'type':'string','enum':['alive','dead','absent','uncertain']},
            'shot_result':{'type':'string','enum':['hit','miss','uncertain','not_applicable']},
            'guidance_followed':{'type':'string','enum':['yes','no','uncertain']},
            'confidence':{'type':'number','minimum':0,'maximum':1}},
        'required':['status','shot_result','guidance_followed','confidence'],
        'additionalProperties':False}}
    text=("Audit the generated video, not the intended story. Images are chronological: "
          "one BEFORE block, then three within/at the END of the block. Track the same enemy. "
          "Return only JSON with status, shot_result, guidance_followed, confidence. "
          "Do not output evidence or free-text explanations. This audits the just-completed "
          "block against the guidance that was used to generate it, NOT a future block. "
          "Report enemy status at the END: alive if visibly living/upright or moving; dead only "
          "with clear temporal evidence of death and a terminal corpse/death state. Falling, "
          "blur, shadows, occlusion or disappearance alone do not prove death. absent means "
          "not visible, not dead. Use uncertain when ambiguous. For shot_result require visible "
          "contact or damage reaction, NOT merely a target under a reticle or muzzle flash. "
          "A dead enemy and a missed shot can coexist. Do not guess HP. Compare guidance with "
          "what actually happened; guidance is an instruction, NOT evidence. "
          f"Player fired in this block: {fired}. Guidance: {guidance}")
    content=[{'type':'text','text':text}]
    for label,img in [('BEFORE',before)]+[(f'AFTER {i}',v) for i,v in enumerate(after)]:
        content.extend([{'type':'text','text':label},{'type':'image_url','image_url':{'url':observer._image_data_url(img)}}])
    r=requests.post(f'{observer.base_url}/chat/completions',json={
        'model':observer.model,'messages':[{'role':'user','content':content}],
        'temperature':0.,'max_tokens':observer.max_tokens,
        'response_format':{'type':'json_schema','json_schema':schema}},timeout=observer.timeout_seconds)
    r.raise_for_status()
    raw=r.json()['choices'][0]['message']['content'];o=json.loads(raw)
    for k in ['status','shot_result','guidance_followed']:
        if o[k] not in schema['schema']['properties'][k]['enum']:raise ValueError(o)
    if not math.isfinite(o['confidence']) or not 0<=o['confidence']<=1:raise ValueError(o)
                                                                              
    lower=guidance.lower()
    expected=('dead' if 'opponent is killed' in lower or 'motionless corpse throughout' in lower
              else 'absent' if 'no opponent remains visible' in lower
              else 'alive' if 'opponent remains alive' in lower or 'same opponent reappears' in lower
              else None)
    o['reported_guidance_followed']=o['guidance_followed']
    if expected is not None and o['status']!='uncertain' and o['status']!=expected:
        o['guidance_followed']='no'
    o['compliance_vote_corrected']=o['reported_guidance_followed']!=o['guidance_followed']
    o['raw_response']=raw
    return o
