''
from __future__ import annotations
import argparse, ast, importlib.util, json, os, sys
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "RIGen"
RUNTIME = ROOT / "inference" / "runtime"
EVALUATOR = RUNTIME / "internal" / "rigen_ar_runner.py"
EVALUATOR_DIR = EVALUATOR.parent
RIGEN_PLAN = ROOT / "configs"
CHECKPOINT = ROOT / "weights" / "model.safetensors"
CONFIG = ROOT / "configs" / "rigen_ar_config.yaml"
for path in (RUNTIME, EVALUATOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from rigen_prompt import compose_runtime_prompt, static_prompt


def apply_transformers5_gemma_shims() -> None:
    ''
    import transformers
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

                                                                    
                                                                           
    if int(transformers.__version__.split('.')[0]) >= 5 and not hasattr(SiglipVisionModel, 'vision_model'):
        SiglipVisionModel.vision_model = property(lambda self: self)
    if not hasattr(Gemma3TextModel, 'rotary_emb_local'):
        Gemma3TextModel.rotary_emb_local = property(lambda self: self.rotary_emb)


def parse_args():
 p=argparse.ArgumentParser(description=f'Run {MODEL_NAME} autoregressive inference.');p.add_argument('--case-id',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--rollout-blocks',type=int,default=6);p.add_argument('--seed',type=int,default=2026092401,help='Optional random seed; defaults to a reproducible release seed.');p.add_argument('--initial-enemy-hp',type=int,choices=(50,100,200),default=100);p.add_argument('--num-inference-steps',type=int,default=30);p.add_argument('--refiner-sigma-start',type=float,choices=(0.5,0.725),default=0.725,help='Refiner start sigma; 0.5 selects the two-step [0.5, 0.25, 0] schedule.');p.add_argument('--vlm-base-url',default='http://127.0.0.1:8012/v1');p.add_argument('--vlm-model',default='internvl3-8b');p.add_argument('--no-visual-feedback',action='store_true');p.add_argument('--vlm-player-controller',dest='vlm_player_controller',action='store_true',default=True);p.add_argument('--no-player-controller',dest='vlm_player_controller',action='store_false');p.add_argument('--action-file',type=Path,default=None);p.add_argument('--release-kill-smoke',action='store_true');return p.parse_args()
def load_evaluator():
 if str(EVALUATOR_DIR) not in sys.path:sys.path.insert(0,str(EVALUATOR_DIR))
 spec=importlib.util.spec_from_file_location('rigen_v11_semantic_evaluator',EVALUATOR)
 if spec is None or spec.loader is None:raise RuntimeError(EVALUATOR)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def write_action_json(path: Path, *, case_id: str, blocks: list[dict]) -> None:
    payload = {
        'schema_version': 'rigen_action_blocks_v1',
        'case_id': case_id,
        'blocks': blocks,
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare_prompt_context_cache(prompt: str, output: Path) -> Path:
    """Encode the refiner prompt for this run and keep the cache with its output."""
    import torch
    import rigen_refine

    if not torch.cuda.is_available():
        raise RuntimeError("RIGen inference requires CUDA to build prompt contexts")
    contexts = rigen_refine.encode_prompt_contexts(
        prompts=[prompt],
        base_model=rigen_refine.BASE_MODEL,
        gemma_root=rigen_refine.GEMMA_ROOT,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    torch.save(
        {key: value.detach().to(device="cpu", dtype=torch.bfloat16) for key, value in contexts.items()},
        output,
    )
    return output


def run_refiner(source: Path, output: Path | None = None, *, sigma_start: float = .725) -> Path:
    """Run the release refiner immediately after the native AR rollout."""
    import torch
    import rigen_refine as helpers
    import rigen_refine_pipeline as refiner

    output = output or source / "refined"
    if output.exists():
        raise FileExistsError(output)
    row = json.loads((source / "closed_loop_manifest.jsonl").read_text())
    context_cache = Path(row["vlm_observation_refiner"]["context_cache"])
    if not context_cache.is_file():
        context_cache = source / context_cache.name
    contexts = torch.load(context_cache, map_location="cpu", weights_only=True)
    original_encode = helpers.encode_prompt_contexts

    def encode(*, prompts, base_model, gemma_root, dtype, device):
        if all(prompt in contexts for prompt in prompts):
            return {
                prompt: contexts[prompt].to(device=device, dtype=dtype)
                for prompt in prompts
            }
        return original_encode(
            prompts=prompts,
            base_model=base_model,
            gemma_root=gemma_root,
            dtype=dtype,
            device=device,
        )

    helpers.encode_prompt_contexts = encode
    refiner.worker(
        source.name,
        sigma_start=sigma_start,
        input_directory=source,
        output_directory=output,
    )
    return output
def install_semantic_main(module,chain,*,visual_feedback=True,vlm_base_url="http://127.0.0.1:8012/v1",vlm_model="internvl3-8b",player_controller=False,custom_actions=None,static_context=None,release_kill_smoke=False,action_output=None,action_case_id=None):
 if player_controller and custom_actions is None and not visual_feedback:
  raise ValueError('--vlm-player-controller requires visual feedback so it can observe the current player view')
 visual_state={"last_image":None,"before_image":None,"observer":None,"scene_observer":None,"lookup":None,"records":[],"player_action_records":[],"action_blocks":[],"navigation_images":[],"navigation_indices":[],"controller_state":{"escape_blocks_remaining":0,"escape_side":None,"forward_streak":0,"last_enemy_side":None,"invisible_blocks":0}}
 if visual_feedback:
  from ltx_trainer.vlm_agent import OpenAIVLMObserver,JoystickPixelLookup
  observer=OpenAIVLMObserver(base_url=vlm_base_url,model=vlm_model)
  observer.healthcheck()
  visual_state["scene_observer"] = __import__("rigen_semantic_core", fromlist=["VLMSceneObserver"]).VLMSceneObserver(observer)
  visual_state["observer"]=observer
  lookup=JoystickPixelLookup.from_path(ROOT/'configs'/'pixel_lookup.json')
  lookup.validate_geometry(width=832,height=480,fps=20)
  visual_state["lookup"]=lookup
                                                                             
                                                                            
                                       
                                                                        
                                                                              
                                     
  chain.configure_vlm_planner(observer, force_hold_low_health=False)
 from rigen_semantic_core import Feedback,SceneObservation
 tree=ast.parse(EVALUATOR.read_text());orig=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');fn=ast.fix_missing_locations(ast.parse(ast.unparse(orig)).body[0])
 fn.body=[n for i,n in enumerate(fn.body) if i!=2]
 loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='target_block')
 before=ast.parse('_semantic_before(locals())').body[0];after=ast.parse('_semantic_after(locals())').body[0];new=[];ib=ia=False
 for node in loop.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='local_prompts' for t in node.targets):new.append(before);ib=True
  new.append(node)
  if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='append' and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='output_audio_tokens':new.append(after);ia=True
 if not(ib and ia):raise RuntimeError((ib,ia))
 loop.body=new
 state={};original_encode=module.encode_prompt_bank
 def capture_encode(**kw):
  s=kw['sampler'];state['text_encoder']=s._text_encoder;state['embeddings_processor']=s._embeddings_processor;return original_encode(**kw)
 module.encode_prompt_bank=capture_encode
 def hook_before(c):
  block=int(c['target_block']);start=int(c['interval_start_frame']);end=int(c['interval_end_frame'])+1
  if visual_feedback:
   visual_state['before_image'] = visual_state['last_image'] or module.pixels_frame_to_image(c['sink_pixels'],0)
   if not visual_state['navigation_images']:
    visual_state['navigation_images']=[visual_state['before_image']]
    visual_state['navigation_indices']=[0]
  axes=c['runtime_joystick'][start:end].tolist();buttons=c['runtime_buttons'][start:end].tolist();manifest_axes=[list(row) for row in axes];row=c.get('row',{});sched=row.get('semantic_observation_schedule',[])
  player_action=None
  chain._advance_enemy_lifecycle(block)
  if custom_actions is not None:
   action=custom_actions[block % len(custom_actions)]
   axes=[list(row) for row in action['axes']]
   buttons=[list(row) for row in action['buttons']]
  elif player_controller:
   from ltx_trainer.vlm_agent import plan_fps_block_actions
   controller_state=visual_state['controller_state']
   images=visual_state['navigation_images'][-2:] or [visual_state['before_image']]
   indices=visual_state['navigation_indices'][-len(images):] or [0]
   navigation=visual_state['observer'].observe_navigation(
    images=images,frame_indices=indices,controller_v2=True,
    verified_enemy_state={**chain.state.to_dict(),'phase':chain.enemy_phase,
                          'respawn_cycle':chain.respawn_cycle})
   player_action=plan_fps_block_actions(
    navigation,move_magnitude=0.15,strafe_magnitude=0.15,
    turn_magnitude=0.012,aim_magnitude=0.0095,vertical_aim_magnitude=0.0045,
    fallback_turn='left' if block%2==0 else 'right')
   phase=chain.enemy_phase
   collision_risk=navigation.path_ahead in {'blocked','corner','uncertain'}
   preventive_reorientation=False
   if collision_risk:
    if player_action.movement in {'strafe_left','strafe_right'}:
     controller_state['escape_side']='left' if player_action.movement=='strafe_left' else 'right'
    elif player_action.turn in {'left','right'}:
     controller_state['escape_side']=player_action.turn
    elif controller_state['escape_side'] is None:
     controller_state['escape_side']='left' if block%2==0 else 'right'
    controller_state['escape_blocks_remaining']=2
   elif controller_state['escape_blocks_remaining']<=0:
    controller_state['forward_streak']=controller_state['forward_streak']+1 if player_action.movement=='forward' else 0
    if controller_state['forward_streak']>3:
     preventive_reorientation=True
     controller_state['escape_side']='left' if block%2==0 else 'right'
     controller_state['escape_blocks_remaining']=1
   escape_applied=bool(collision_risk or controller_state['escape_blocks_remaining']>0 or preventive_reorientation)
   if escape_applied:
    side=controller_state['escape_side'] or ('left' if block%2==0 else 'right')
    sign=-1.0 if side=='left' else 1.0
    player_action=replace(player_action,left_x=sign*0.15,left_y=0.06,
                          right_x=-0.020 if side=='left' else 0.020,right_y=0.0,
                          movement='strafe_left' if side=='left' else 'strafe_right',turn=side)
    controller_state['forward_streak']=0
    controller_state['escape_blocks_remaining']=max(0,controller_state['escape_blocks_remaining']-1)
   aim_solution=None;alignment=None
   if navigation.enemy_visible:
    controller_state['invisible_blocks']=0
    if navigation.enemy_x_norm is None or navigation.enemy_y_norm is None:
     raise RuntimeError('RIGen controller requires enemy coordinates')
    controller_state['last_enemy_side']='left' if navigation.enemy_x_norm<0.5 else 'right'
    if phase=='active':
     alignment=visual_state['observer'].observe_aim_target(image=images[-1],crop_size=320)
    aligned=bool(alignment is not None and alignment.aim_target=='enemy' and alignment.confidence>=0.55)
    if aligned:
     player_action=replace(player_action,right_x=0.0,right_y=0.0)
     aim_solution={'mode':'hold_verified_current_alignment','target_x_norm':navigation.enemy_x_norm,'target_y_norm':navigation.enemy_y_norm}
    else:
     solved_x,solved_y,shift=visual_state['lookup'].solve_right_stick_for_target(
      target_x_norm=navigation.enemy_x_norm,target_y_norm=navigation.enemy_y_norm,
      left_x=player_action.left_x,left_y=player_action.left_y,num_action_frames=3,
      gain=0.90,max_right_x=0.08,max_right_y=0.05)
     player_action=replace(player_action,right_x=solved_x,right_y=solved_y)
     aim_solution={'mode':'coordinate_lookup','target_x_norm':navigation.enemy_x_norm,'target_y_norm':navigation.enemy_y_norm,'projected_shift':shift.to_dict()}
   else:
    controller_state['invisible_blocks']+=1
    if controller_state['last_enemy_side'] is None: search_left=(controller_state['invisible_blocks']//2)%2==0
    elif controller_state['invisible_blocks']<=2: search_left=controller_state['last_enemy_side']=='left'
    else:
     base_left=controller_state['last_enemy_side']=='left';search_left=base_left if ((controller_state['invisible_blocks']-3)//2)%2==0 else not base_left
    player_action=replace(player_action,right_x=-0.020 if search_left else 0.020,right_y=0.0,turn='left' if search_left else 'right')
   axes=[[player_action.left_x,player_action.left_y,player_action.right_x,player_action.right_y] for _ in axes]
   buttons=[list(row) for row in buttons]
   for row in buttons: row[0]=0.0
   fire_frames=[]
   if player_action.fire and phase=='active' and navigation.enemy_visible and block != 0:
    fire_frame=min(len(buttons)-1,2);buttons[fire_frame][0]=1.0;fire_frames.append(fire_frame)
   else:
    player_action=replace(player_action,fire=False)
   record={'block':block,'lifecycle_phase':phase,'respawn_cycle':chain.respawn_cycle,'navigation':navigation.to_dict(),'action_plan':player_action.to_dict(),'manifest_axes':manifest_axes,'controller_axes':axes,'fire_requested_by_controller':bool(player_action.fire),'fire_applied_frames':fire_frames,'player_fire_allowed_by_rigen':phase=='active','enemy_fire_allowed_by_rigen':False,'navigation_safety':{'collision_risk':collision_risk,'preventive_reorientation':preventive_reorientation,'escape_applied':escape_applied,'escape_side':controller_state['escape_side'],'escape_blocks_remaining':controller_state['escape_blocks_remaining'],'forward_streak':controller_state['forward_streak']},'aim_solution':aim_solution}
   visual_state['player_action_records'].append(record)
   if chain.output_dir is not None:
    with (chain.output_dir/'semantic_player_action_trace.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(record,ensure_ascii=False)+'\n')
  value=(visual_state['scene_observer'].observe(visual_state['before_image']) if visual_feedback else (sched[block] if isinstance(sched,list) and block<len(sched) else {}))
  proposal=chain.before_block(block=block,observation=SceneObservation.from_value(value),axes=axes,buttons=buttons,image=visual_state['before_image'])
  if action_output is not None:
   entry={'block_index':block,'axes':[[float(value) for value in row] for row in proposal.gate.axes],'buttons':[[float(value) for value in row] for row in proposal.gate.buttons]}
   visual_state['action_blocks']=[item for item in visual_state['action_blocks'] if item['block_index']!=block]
   visual_state['action_blocks'].append(entry)
   visual_state['action_blocks'].sort(key=lambda item:item['block_index'])
   write_action_json(action_output,case_id=action_case_id or action_output.parent.name,blocks=visual_state['action_blocks'])
  import torch
  c['runtime_joystick'][start:end]=torch.as_tensor(proposal.gate.axes,dtype=c['runtime_joystick'].dtype,device=c['runtime_joystick'].device);c['runtime_buttons'][start:end]=torch.as_tensor(proposal.gate.buttons,dtype=c['runtime_buttons'].dtype,device=c['runtime_buttons'].device)
  source_prompt=str(c['runtime_prompt_schedule'][block]);prompt=compose_runtime_prompt(source_prompt,proposal.guidance,fallback_static=static_context);c['runtime_prompt_schedule'][block]=prompt
  if chain.output_dir is not None:
   with (chain.output_dir/'semantic_prompt_trace.jsonl').open('a',encoding='utf-8') as handle:
    canonical_prompt=(f"{static_context}\n[CURRENT OPPONENT MOTION AND OUTCOME]\n{proposal.guidance}" if static_context is not None else prompt)
    handle.write(json.dumps({'block':block,'source_prompt':source_prompt,'canonical_training_prompt':canonical_prompt,'rendered_prompt':prompt,'guidance':proposal.guidance,'dynamic_section_count':canonical_prompt.count('[CURRENT OPPONENT')},ensure_ascii=False)+'\n')
  text_encoder=state.get('text_encoder');processor=state.get('embeddings_processor')
  if text_encoder is None or processor is None:raise RuntimeError('prompt encoder was not captured')
  s=c['sampler'];s._text_encoder=text_encoder;s._embeddings_processor=processor
  fresh=original_encode(sampler=s,prompts=[prompt],device=c['device'],dtype=c['component_dtype'],offload_after_encode=False);bank=c['prompt_bank'];bank.update(fresh);length=max(v[0].shape[0] for v in bank.values())
  import torch.nn.functional as F
  for key,(video,audio,mask) in list(bank.items()):bank[key]=(F.pad(video,(0,0,0,length-video.shape[0])),F.pad(audio,(0,0,0,length-audio.shape[0])),F.pad(mask,(0,length-mask.shape[0])))
  s._text_encoder=None;s._embeddings_processor=None
 def hook_after(c):
  if not chain.proposals:raise RuntimeError('semantic after without proposal')
  proposal=chain.proposals[-1]
  if visual_feedback:
   from ltx_trainer.vlm_video_authoritative_v11 import audit_block
                                                                          
                                                                               
   tokens=__import__('torch').cat(c['output_video_tokens'],dim=1)
   prefix_frames=1+16*(len(c['output_video_tokens'])-1)
   preview=c['sampler']._decode_video_latent(tokens.to(c['component_dtype']),replace(c['preview_config'],num_frames=prefix_frames),c['device'])
   block=int(c['target_block']);frames=[module.tensor_frame_to_image(preview,prefix_frames-16+i) for i in range(16)]
   fire_offsets=list(proposal.gate.fire_offsets)
   fire_frames=[(offset, frames[offset]) for offset in fire_offsets]
                                                                            
                                                                           
                                                                     
                              
   audit_mode='corpse' if not proposal.planned.enemy_alive else 'active_left'
   result=audit_block(visual_state['observer'],reference=c['enemy_reference'],before=visual_state['before_image'],fire_frames=fire_frames,late=frames[-3],end=frames[-1],prompt_mode=audit_mode)
   targets=[result.get(f'frame_{offset}') for offset in fire_offsets]
   player_hits=tuple(offset for offset,target in zip(fire_offsets,targets,strict=True) if target=='enemy')
   status=result['status']
   if player_hits and proposal.prior.enemy_hp <= chain.rules.config.player_hit_damage:
    status='dead'
   feedback=Feedback(enemy_status={'alive':'alive','dead':'dead','not_visible':'not_visible'}[status],player_hits=player_hits)
   previous=visual_state['navigation_images'][-1] if visual_state['navigation_images'] else visual_state['before_image']
   visual_state['last_image']=frames[-1]
   visual_state['navigation_images']=[previous,frames[-1]]
   visual_state['navigation_indices']=[max(0,16*block),16*(block+1)]
   visual_state['records'].append({'block':block,'status':status,'feedback':feedback.to_dict()})
   visual_state['records'][-1]['audit']=result
   (chain.output_dir/'semantic_visual_feedback.jsonl').open('a',encoding='utf-8').write(json.dumps(visual_state['records'][-1])+'\n')
   del preview,tokens
  else: feedback=Feedback()
  chain.after_block(proposal,feedback)
 ns=module.__dict__;ns.update(_semantic_before=hook_before,_semantic_after=hook_after);compiled=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));exec(compile(compiled,'<semantic-rigen-v11>','exec'),ns);module.main=ns['main']

def main():
 a=parse_args()
 a.output_dir=a.output_dir.resolve()
 if a.action_file is not None:a.action_file=a.action_file.resolve()
 os.chdir(ROOT)
 apply_transformers5_gemma_shims()
 if a.rollout_blocks<1:raise ValueError('--rollout-blocks must be positive')
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 a.output_dir.mkdir(parents=True)
 action_output=a.output_dir/'actions.json'
 write_action_json(action_output,case_id=a.case_id,blocks=[])
 source=ROOT/'cases'/a.case_id/'manifest.jsonl'
 if not source.is_file():raise FileNotFoundError(source)
 row=json.loads(source.read_text().splitlines()[0]);schedule=list(row.get('prompt_schedule',[]))
 if not schedule:raise ValueError('manifest has no prompt schedule')
 row['prompt_schedule']=[schedule[i%len(schedule)] for i in range(a.rollout_blocks)]
 if isinstance(row.get('active_ref_schedule'),list) and row['active_ref_schedule']:
  active=list(row['active_ref_schedule']);row['active_ref_schedule']=[active[i%len(active)] for i in range(a.rollout_blocks)]
 context_cache=prepare_prompt_context_cache(static_prompt(row),a.output_dir/'prompt_contexts.pt')
 row['vlm_observation_refiner']['context_cache']=str(context_cache)
 manifest=a.output_dir/'closed_loop_manifest.jsonl';manifest.write_text(json.dumps(row)+'\n')
 from rigen_loop import RIGenAREnemyChain, RIGenAREnemyPlanner
 from rigen_semantic_core import CombatStateEngine,RuleConfig
 planner=RIGenAREnemyPlanner(force_hold_low_health=False)
 rules=CombatStateEngine(RuleConfig(enemy_retreat_threshold=50,allow_enemy_approach=False))
 chain=RIGenAREnemyChain(output_dir=a.output_dir/'semantic_chain',planner=planner,rules=rules,initial_enemy_hp=a.initial_enemy_hp,respawn_enabled=False);chain.output_dir.mkdir(parents=True,exist_ok=True);evaluator=load_evaluator();custom_actions=None
 if a.action_file is not None:
  payload=json.loads(a.action_file.read_text());custom_actions=payload['blocks'] if isinstance(payload,dict) else payload
  if not custom_actions: raise ValueError('action file has no blocks')
 install_semantic_main(evaluator,chain,visual_feedback=not a.no_visual_feedback,vlm_base_url=a.vlm_base_url,vlm_model=a.vlm_model,player_controller=a.vlm_player_controller,custom_actions=custom_actions,static_context=static_prompt(row),release_kill_smoke=a.release_kill_smoke,action_output=action_output,action_case_id=a.case_id)
 sys.argv=[str(EVALUATOR),'--config',str(CONFIG),'--checkpoint',str(CHECKPOINT),'--manifest',str(manifest),'--data-root',str(ROOT),'--manifest-index','0','--first-target-block','0','--bootstrap-from-empty-history','--rollout-blocks',str(a.rollout_blocks),'--seed',str(a.seed),'--num-inference-steps',str(a.num_inference_steps),'--denoising-update-mode','euler','--scheduler-timestep-shift','2.05','--guidance-mode','conditional_only','--skip-ground-truth','--save-refiner-cache','--low-host-memory','--device','cuda:0','--dtype','bf16','--native-stage2-ar','--condition-image-only','--output-dir',str(a.output_dir/'model_rollout')]
 evaluator.main()
 run_refiner(a.output_dir, sigma_start=a.refiner_sigma_start)
 contract={'schema_version':'semantic_closed_loop_run_v3','case_id':a.case_id,'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'config':str(CONFIG.relative_to(ROOT)),'manifest':str(manifest),'rollout_blocks':a.rollout_blocks,'num_inference_steps':a.num_inference_steps,'refiner_sigma_start':a.refiner_sigma_start,'refiner_sigma_schedule':([.5,.25,0.] if a.refiner_sigma_start==.5 else [.725,.421875,0.]),'action_json':str(action_output),'prompt_context_cache':str(context_cache),'condition_image_only':True,'semantic_chain_trace':str(a.output_dir/'semantic_chain/semantic_rigen_trace.json'),'semantic_prompt_trace':str(a.output_dir/'semantic_chain/semantic_prompt_trace.jsonl'),'semantic_player_action_trace':str(a.output_dir/'semantic_chain/semantic_player_action_trace.jsonl') if a.vlm_player_controller else None,'prompt_guidance_injected_into_model':True,'prompt_composer':'rigen_prompt.compose_runtime_prompt','prompt_dynamic_section':'[CURRENT OPPONENT MOTION AND OUTCOME]','player_planner_calls':a.rollout_blocks if a.vlm_player_controller else 0,'player_action_source':('vlm_closed_loop_controller_v2' if a.vlm_player_controller else 'custom_action_json' if a.action_file is not None else 'zero_action_window'),'player_fire_allowed':True,'enemy_fire_allowed':False,'enemy_planner':'vlm_enemy_behavior_with_python_fallback' if not a.no_visual_feedback else 'python_fallback_no_visual_observation','feedback_mode':'vlm_visual_state_audit_with_fire_audit' if not a.no_visual_feedback else 'explicit_unknown_after_block','action_controller':('continuous_coordinate_v2_then_semantic_gate' if a.vlm_player_controller else 'semantic_gate_over_action_json' if a.action_file is not None else 'semantic_gate_over_zero_action_window'),'enemy_lifecycle':'dead_until_end_of_rollout','respawn_enabled':False,'release_kill_smoke':a.release_kill_smoke,'initial_enemy_hp':a.initial_enemy_hp,'enemy_retreat_threshold':rules.config.enemy_retreat_threshold,'enemy_approach_allowed':rules.config.allow_enemy_approach,'audit_decode':'full_prefix_then_current_block','source_evaluator_unchanged':True,'vlm_visual_feedback':not a.no_visual_feedback,'vlm_base_url':a.vlm_base_url,'vlm_model':a.vlm_model,'refiner_inline':True,'refiner_output':str(a.output_dir/'refined'/'refined.mp4'),'block1_fire_suppressed':bool(a.vlm_player_controller)}
 contract['model_name']=MODEL_NAME
 (a.output_dir/'test_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
if __name__=='__main__':main()
