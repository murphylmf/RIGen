''
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('.')
SOURCE = ROOT/'outputs/eval_stage2long_slref_synthetic10_vlm_20260910_v1'
OUT = ROOT/'outputs/eval_stage2long_synthetic10_refiner07_20260911_v1'
REVIEW = OUT/'review'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def scene_weapon_prompt(text):
    ''
    headers = ('[ACTIVE OPPONENT IDENTITY]', '[TRACKED ENEMY IDENTITY]')
    boundaries = [text.index(header) for header in headers if header in text]
    if not boundaries:
        raise ValueError('Missing supported enemy identity header in refiner prompt')
    prompt = text[:min(boundaries)].strip()
    if '[ACTIVE WEAPON]' not in prompt or '[CURRENT' in prompt:
        raise ValueError('Refiner prompt must contain scene/weapon only')
    return prompt


def write(path, payload):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def refiner_schedule(sigma_start: float = .725) -> list[float]:
    ''
    if sigma_start == .5:
        return [.5, .25, 0.]
    if sigma_start == .725:
        return [.725, .421875, 0.]
    if sigma_start == .421875:
                                                                            
        return [.421875, 0.]
    if sigma_start == .909375:
        return [.909375, .725, .421875, 0.]
    raise ValueError(f'Unsupported refiner start sigma: {sigma_start}')


def worker(name, *, sigma_start: float = .725, shared_models: dict | None = None,
           prompt_override: str | None = None, initial_refined_latent: Path | None = None,
           noise_seed: int | None = None, input_directory: Path | None = None,
           output_directory: Path | None = None):
    schedule = refiner_schedule(sigma_start)
    if initial_refined_latent is not None and noise_seed is None:
        raise ValueError('Second-pass refinement requires an explicit new noise seed')
    if prompt_override is not None and not prompt_override.strip():
        raise ValueError('Empty prompt override')
    import torch
    from rigen_refine import (
        BASE_MODEL, GEMMA_ROOT, DISTILLED_LORA, SPATIAL_UPSAMPLER,
        encode_prompt_contexts, VideoUpsampler, ImageConditioner,
        ImageConditioningInput, combined_image_conditionings,
        CachedOriginalDiffusionStage, LoraPathStrengthAndSDOps,
        LTXV_LORA_COMFY_RENAMING_MAP, VideoConditionByLatentIndex,
        SimpleDenoiserNoCamTokens, ModalitySpec, FixedVideoNoiser,
        TilingConfig, SpatialTilingConfig, TemporalTilingConfig, save_video)
    from rigen_decode import _decode
    base = Path(output_directory) if output_directory is not None else OUT/name
    base.mkdir(parents=True, exist_ok=True)
    source = Path(input_directory) if input_directory is not None else SOURCE/name
    cache_path = source/('model_rollout/refiner_cache.pt' if input_directory is not None else 'closed_loop/refiner_cache.pt')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    assert cache['schema_version']=='ar_sliding_ref_refiner_cache_v1'
    # A release run may be moved after generation.  Keep the refiner
    # self-contained by resolving cached media relative to the run when an
    # older absolute path is no longer available.
    for key in ('condition_image_path', 'source_raw_video'):
        cached_path = Path(cache[key])
        if not cached_path.is_file():
            local_path = source / 'model_rollout' / cached_path.name
            if local_path.is_file():
                cache[key] = str(local_path)
            else:
                raise FileNotFoundError(cache[key])
    frames=int(cache['geometry']['num_frames'])
    assert frames>1 and (frames-1)%16==0 and cache['geometry']['fps']==20
    total_latents=(frames-1)//8+1
    assert tuple(cache['video_latent'].shape)==(1,128,total_latents,15,26)
    assert len(cache['prompt_schedule'])==(frames-1)//16
    device=torch.device('cuda:0');dtype=torch.bfloat16
    def stage(value):
        print(f'[{stamp()}] {name}: {value}',flush=True)
        write(base/'progress.json',dict(stage=value,utc=stamp()))
                                                                             
    manifest=json.loads((source/'closed_loop_manifest.jsonl').read_text())
    prompt=scene_weapon_prompt(manifest['prompt_text']) if prompt_override is None else prompt_override
    stage('encode_scene_weapon_prompt')
    context=encode_prompt_contexts(prompts=[prompt],base_model=BASE_MODEL,
        gemma_root=GEMMA_ROOT,dtype=dtype,device=device)[prompt].to(device)
    if context.ndim==2:context=context.unsqueeze(0)
    stage('full_clip_upsample' if initial_refined_latent is None else 'load_prior_refined_highres_latent')
    with torch.inference_mode():
        if initial_refined_latent is None:
            owner=VideoUpsampler(str(BASE_MODEL),str(SPATIAL_UPSAMPLER),dtype,device)
            up=owner(cache['video_latent'].to(device=device,dtype=dtype)).detach()
            del owner
        else:
                                                                                     
            up=torch.load(Path(initial_refined_latent),map_location='cpu',weights_only=True).to(device=device,dtype=dtype)
        assert tuple(up.shape)==(1,128,total_latents,30,52)
        generator=torch.Generator(device=device)
        if noise_seed is None:
            generator.set_state(cache['generator_state'].cpu())
        else:
            generator.manual_seed(noise_seed)
        noise=torch.randn(up.shape,device=device,dtype=dtype,generator=generator)
        def tensor_hash(value):
            return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        input_key='upsampled_latent' if initial_refined_latent is None else 'initial_refined_latent'
        input_hashes=dict(raw_latent=tensor_hash(cache['video_latent']),
                          **{input_key:tensor_hash(up)}, noise=tensor_hash(noise))
        owner=ImageConditioner(str(BASE_MODEL),dtype,device)
        initial_conditions=owner(lambda encoder:combined_image_conditionings(
            images=[ImageConditioningInput(path=cache['condition_image_path'],
                    frame_idx=0,strength=1.0,crf=33)],height=960,width=1664,
            video_encoder=encoder,dtype=dtype,device=device))
        del owner
        gc.collect();torch.cuda.empty_cache()
        loras=(LoraPathStrengthAndSDOps(str(DISTILLED_LORA),1.0,LTXV_LORA_COMFY_RENAMING_MAP),)
        model_key=(str(BASE_MODEL), str(DISTILLED_LORA), str(dtype), str(device))
        model_reused=shared_models is not None and model_key in shared_models
        if model_reused:
            model=shared_models[model_key]
        else:
            model=CachedOriginalDiffusionStage(str(BASE_MODEL),dtype,device,loras=loras)
            if shared_models is not None:
                shared_models[model_key]=model
        sigmas=torch.tensor(schedule,device=device,dtype=torch.float32)
        committed=None;records=[]
        windows=[(0,min(21,total_latents))]
        while windows[-1][1]<total_latents:
            previous_end=windows[-1][1]
            windows.append((previous_end-3,min(previous_end+20,total_latents)))
        for i,(start,end) in enumerate(windows):
            stage(f'refine_window_{i}_{start}_{end}')
            before=time.perf_counter()
            bridge=None if i==0 else committed[:,:,start:start+3].clone()
            conditions=initial_conditions if bridge is None else [
                VideoConditionByLatentIndex(latent=bridge,strength=1.0,latent_idx=0)]
            refined,_=model(denoiser=SimpleDenoiserNoCamTokens(v_context=context,a_context=None),
                sigmas=sigmas,noiser=FixedVideoNoiser(noise[:,:,start:end]),
                width=1664,height=960,frames=1+(end-start-1)*8,fps=20.,
                video=ModalitySpec(context=context,conditionings=conditions,
                    noise_scale=schedule[0],initial_latent=up[:,:,start:end]),audio=None)
            assert refined is not None
            latent=refined.latent
            assert tuple(latent.shape)==(1,128,end-start,30,52)
            exact=bridge is None or torch.equal(bridge,latent[:,:,:3])
            if not exact:raise RuntimeError('Refined handoff prefix changed')
            old=committed
            committed=latent.clone() if i==0 else torch.cat((committed[:,:,:start+2],latent[:,:,2:]),dim=2)
            if old is not None and not torch.equal(old,committed[:,:,:old.shape[2]]):
                raise RuntimeError('Previously committed refined prefix changed')
            torch.save(latent.cpu(),base/f'window_{i}_latent.pt')
            records.append(dict(start=start,end=end,prefix_exact=exact,seconds=time.perf_counter()-before))
            write(base/'windows.json',records)
            del old,refined,latent
        assert tuple(committed.shape)==(1,128,total_latents,30,52)
        output_latent_hash=tensor_hash(committed)
        torch.save(committed.cpu(),base/'refined_latent.pt')
                                                                          
        if shared_models is None:
            model._cached_transformer=None
        del model,up,noise,context,initial_conditions,conditions,bridge
        gc.collect();torch.cuda.empty_cache()
        stage('single_full_clip_decode')
        video,seconds=_decode(committed,generator_state=cache['generator_state'],
            tiling=TilingConfig(spatial_config=SpatialTilingConfig(512,64),
                temporal_config=TemporalTilingConfig(((frames+7)//8)*8,0)),
            model=BASE_MODEL,dtype=dtype,device=device)
        assert tuple(video.shape)==(3,frames,960,1664)
        save_video(video,base/'silent.mp4',fps=20)
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(base/'silent.mp4'),
        '-i',cache['source_raw_video'],'-map','0:v:0','-map','1:a:0?',
        '-c:v','copy','-c:a','aac','-movflags','+faststart',str(base/'refined.mp4')],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(base/'refined.mp4'),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(base/'refined.mp4')]))
    v=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (int(v['width']),int(v['height']),int(v['nb_frames']))==(1664,960,frames)
    write(base/'run_info.json',dict(source_cache=str(cache_path),base_model=str(BASE_MODEL),
        distilled_lora=str(DISTILLED_LORA),upsampler=str(SPATIAL_UPSAMPLER),
        sigmas=schedule,noise_scale=schedule[0],input_hashes=input_hashes,action_conditioned=False,
        transformer_weights_reused=model_reused,
        prompt_policy=('scene and weapon only; no fixed opponent visibility or outcome'
                       if prompt_override is None else 'explicit matched-experiment prompt override'),
        refiner_prompt=prompt,windows=records,full_upsample_once=initial_refined_latent is None,
        input_mode='upsampled_raw' if initial_refined_latent is None else 'prior_refined_highres_latent',
        initial_refined_latent=None if initial_refined_latent is None else str(Path(initial_refined_latent).resolve()),
        noise_seed=noise_seed,output_latent_hash=output_latent_hash,
        single_full_temporal_decode=True,decode_seconds=seconds,
        video_frames=frames,video_size=[1664,960],completed_at=stamp()))
    stage('complete')


def main(resume=False):
    OUT.mkdir(exist_ok=True);REVIEW.mkdir(exist_ok=True)
    if resume:
        prior=json.loads((REVIEW/'status.json').read_text())
        assert prior['status']=='failed'
        for c in prior['cases']:
            if c.get('pid'):
                try:os.kill(c['pid'],0)
                except ProcessLookupError:pass
                else:raise RuntimeError('Prior child still alive')
    else:
        with (OUT/'launch.lock').open('x'):pass
    plan=json.loads((SOURCE/'plan.json').read_text())
    assert json.loads((SOURCE/'complete.json').read_text())['cases']==10
    for name in plan['cases']:
        assert (SOURCE/name/'closed_loop/refiner_cache.pt').is_file()
        assert (SOURCE/name/'complete.json').is_file()
        (OUT/name).mkdir(exist_ok=resume)
        media=REVIEW/name;media.mkdir(exist_ok=resume)
        for source,target in [('pred_raw.mp4','raw.mp4'),('pred_raw_actions.mp4','actions.mp4'),('agent_trace.jsonl','agent_trace.jsonl')]:
            if not (media/target).exists():os.link(SOURCE/name/'closed_loop'/source,media/target)
    state=dict(status='running',started_at=stamp(),cases=[dict(name=n,status='waiting') for n in plan['cases']])
    write(REVIEW/'status.json',state)
    try:
        for c in state['cases']:
            name=c['name'];c.update(status='running',started_at=stamp())
            with (OUT/name/('worker_retry1.log' if resume else 'worker.log')).open('xb') as log:
                child=subprocess.Popen([sys.executable,'-u',__file__,'--case',name],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                c['pid']=child.pid
                while child.poll() is None:
                    p=OUT/name/'progress.json'
                    if p.exists():c['progress']=json.loads(p.read_text())
                    state['heartbeat']=stamp();write(REVIEW/'status.json',state);time.sleep(20)
            c['exit_code']=child.returncode
            if child.returncode:raise RuntimeError(f'{name} failed: {child.returncode}')
            for filename in ['refined.mp4','run_info.json']:
                os.link(OUT/name/filename,REVIEW/name/filename)
            c.update(status='complete',finished_at=stamp())
            write(REVIEW/'status.json',state)
        state.update(status='complete',finished_at=stamp())
        write(REVIEW/'status.json',state)
        with (OUT/'server.log').open('xb') as log:
            server=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/serve_crossfps_ppt_prompt_review.py'),
                '--bundle',str(REVIEW),'--host','0.0.0.0','--port','18941'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'server.json',dict(pid=server.pid,url='http://localhost:18941',utc=stamp()))
    except BaseException as error:
        state.update(status='failed',error=repr(error),heartbeat=stamp())
        write(REVIEW/'status.json',state)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--resume',action='store_true');args=p.parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    worker(args.case) if args.case else main(args.resume)
